// Function: sub_E54220
// Address: 0xe54220
//
_BYTE *__fastcall sub_E54220(__int64 a1, char *a2, __int64 a3, char *a4, size_t a5)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r15
  int v12; // ebx
  __int64 v13; // r8
  _WORD *v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdi
  size_t v17; // rax
  size_t v18; // r9
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  unsigned int v22; // eax
  __int64 v23; // rcx
  unsigned __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  unsigned int dest; // [rsp+24h] [rbp-5Ch] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h] BYREF
  size_t v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h]
  int v31; // [rsp+40h] [rbp-40h]
  __int16 v32; // [rsp+44h] [rbp-3Ch]
  char v33; // [rsp+46h] [rbp-3Ah]

  v28 = 0x206F666E692E09LL;
  sub_904010(*(_QWORD *)(a1 + 304), (const char *)&v28);
  sub_E51560(a1, a2, a3, *(_QWORD *)(a1 + 304));
  sub_904010(*(_QWORD *)(a1 + 304), ", ");
  v9 = *(_QWORD *)(a1 + 304);
  v29 = a5;
  v32 = 1;
  v30 = 0;
  v31 = 10;
  v33 = 1;
  v10 = sub_CB6AF0(v9, (__int64)&v29);
  sub_904010(v10, ", ");
  if ( !a5 )
    return sub_E4D880(a1);
  v24 = 4 * ((a5 - 1) >> 2) + 4;
  if ( a5 > 3 )
  {
    v11 = 4;
    v12 = 0;
    while ( 1 )
    {
      if ( v12 )
      {
        v13 = *(_QWORD *)(a1 + 304);
        --v12;
        v14 = *(_WORD **)(v13 + 32);
        if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 > 1u )
          goto LABEL_5;
      }
      else
      {
        sub_E4D880(a1);
        v25 = *(_QWORD *)(a1 + 304);
        v17 = strlen((const char *)&v28);
        v13 = v25;
        v18 = v17;
        v14 = *(_WORD **)(v25 + 32);
        v19 = *(_QWORD *)(v25 + 24) - (_QWORD)v14;
        if ( v18 > v19 )
        {
          sub_CB6200(v25, (unsigned __int8 *)&v28, v18);
          v13 = *(_QWORD *)(a1 + 304);
          v14 = *(_WORD **)(v13 + 32);
          v19 = *(_QWORD *)(v13 + 24) - (_QWORD)v14;
        }
        else if ( v18 )
        {
          if ( (_DWORD)v18 )
          {
            v22 = 0;
            do
            {
              v23 = v22++;
              *((_BYTE *)v14 + v23) = *((_BYTE *)&v28 + v23);
            }
            while ( v22 < (unsigned int)v18 );
          }
          *(_QWORD *)(v25 + 32) += v18;
          v13 = *(_QWORD *)(a1 + 304);
          v14 = *(_WORD **)(v13 + 32);
          v19 = *(_QWORD *)(v13 + 24) - (_QWORD)v14;
        }
        v12 = 5;
        if ( v19 > 1 )
        {
LABEL_5:
          *v14 = 8236;
          *(_QWORD *)(v13 + 32) += 2LL;
          goto LABEL_6;
        }
      }
      sub_CB6200(v13, (unsigned __int8 *)", ", 2u);
LABEL_6:
      v15 = *(_DWORD *)&a4[v11 - 4];
      v33 = 1;
      v16 = *(_QWORD *)(a1 + 304);
      v30 = 0;
      v31 = 10;
      v29 = _byteswap_ulong(v15);
      v32 = 1;
      sub_CB6AF0(v16, (__int64)&v29);
      if ( a5 < v11 + 4 )
      {
        if ( (_DWORD)a5 != (_DWORD)v24 )
        {
          dest = 0;
          memcpy(&dest, &a4[v11], a5 - v11);
          if ( v12 )
            goto LABEL_15;
          goto LABEL_24;
        }
        return sub_E4D880(a1);
      }
      v11 += 4;
    }
  }
  if ( (_DWORD)a5 != (_DWORD)v24 )
  {
    dest = 0;
    memcpy(&dest, a4, a5);
LABEL_24:
    sub_E4D880(a1);
    sub_904010(*(_QWORD *)(a1 + 304), (const char *)&v28);
LABEL_15:
    sub_904010(*(_QWORD *)(a1 + 304), ", ");
    v33 = 1;
    v20 = *(_QWORD *)(a1 + 304);
    v30 = 0;
    v31 = 10;
    v29 = _byteswap_ulong(dest);
    v32 = 1;
    sub_CB6AF0(v20, (__int64)&v29);
  }
  return sub_E4D880(a1);
}
