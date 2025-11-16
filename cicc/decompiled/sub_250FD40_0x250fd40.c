// Function: sub_250FD40
// Address: 0x250fd40
//
__int64 __fastcall sub_250FD40(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v4)(__int64); // rax
  char v5; // al
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r9
  _BYTE *v9; // rax
  __int64 v10; // r15
  const char *v11; // rax
  size_t v12; // rdx
  _BYTE *v13; // rdi
  unsigned __int8 *v14; // rsi
  _BYTE *v15; // rax
  signed __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  _BYTE *v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-38h]
  size_t v26; // [rsp+8h] [rbp-38h]

  sub_904010(a1, "set-state(< {");
  v4 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL);
  if ( v4 == sub_2505E60 )
    v5 = *(_BYTE *)(a2 + 17);
  else
    v5 = v4(a2);
  if ( v5 )
  {
    v6 = *(_QWORD *)(a2 + 56);
    v7 = v6 + 24LL * *(unsigned int *)(a2 + 64);
    while ( v7 != v6 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)v6;
        if ( **(_BYTE **)v6 )
        {
          sub_A69870(*(_QWORD *)v6, (_BYTE *)a1, 0);
          v21 = *(_BYTE **)(a1 + 32);
          if ( *(_BYTE **)(a1 + 24) == v21 )
          {
            v24 = sub_CB6200(a1, (unsigned __int8 *)"[", 1u);
            v16 = *(unsigned __int8 *)(v6 + 16);
            v17 = v24;
          }
          else
          {
            *v21 = 91;
            v17 = a1;
            ++*(_QWORD *)(a1 + 32);
            v16 = *(unsigned __int8 *)(v6 + 16);
          }
        }
        else
        {
          v9 = *(_BYTE **)(a1 + 32);
          if ( *(_BYTE **)(a1 + 24) == v9 )
          {
            v25 = *(_QWORD *)v6;
            v22 = sub_CB6200(a1, (unsigned __int8 *)"@", 1u);
            v8 = v25;
            v10 = v22;
          }
          else
          {
            *v9 = 64;
            v10 = a1;
            ++*(_QWORD *)(a1 + 32);
          }
          v11 = sub_BD5D20(v8);
          v13 = *(_BYTE **)(v10 + 32);
          v14 = (unsigned __int8 *)v11;
          v15 = *(_BYTE **)(v10 + 24);
          if ( v12 > v15 - v13 )
          {
            v10 = sub_CB6200(v10, v14, v12);
            v15 = *(_BYTE **)(v10 + 24);
            v13 = *(_BYTE **)(v10 + 32);
          }
          else if ( v12 )
          {
            v26 = v12;
            memcpy(v13, v14, v12);
            v23 = (_BYTE *)(*(_QWORD *)(v10 + 32) + v26);
            *(_QWORD *)(v10 + 32) = v23;
            v15 = *(_BYTE **)(v10 + 24);
            v13 = v23;
          }
          if ( v15 == v13 )
          {
            v10 = sub_CB6200(v10, (unsigned __int8 *)"[", 1u);
          }
          else
          {
            *v13 = 91;
            ++*(_QWORD *)(v10 + 32);
          }
          v16 = *(unsigned __int8 *)(v6 + 16);
          v17 = v10;
        }
        v18 = sub_CB59F0(v17, v16);
        v19 = *(_QWORD *)(v18 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v18 + 24) - v19) <= 2 )
          break;
        v6 += 24;
        *(_BYTE *)(v19 + 2) = 32;
        *(_WORD *)v19 = 11357;
        *(_QWORD *)(v18 + 32) += 3LL;
        if ( v7 == v6 )
          goto LABEL_18;
      }
      v6 += 24;
      sub_CB6200(v18, "], ", 3u);
    }
LABEL_18:
    if ( *(_BYTE *)(a2 + 264) )
      sub_904010(a1, "undef ");
  }
  else
  {
    sub_904010(a1, "full-set");
  }
  sub_904010(a1, "} >)");
  return a1;
}
