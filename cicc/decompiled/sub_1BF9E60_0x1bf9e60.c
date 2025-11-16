// Function: sub_1BF9E60
// Address: 0x1bf9e60
//
__int64 __fastcall sub_1BF9E60(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  _BYTE *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // r13d
  unsigned __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rbx
  unsigned int v11; // r13d
  unsigned int v13; // eax
  __int64 v14; // r9
  _QWORD *v15; // rax
  __int64 v17; // [rsp+20h] [rbp-110h]
  __int64 v18; // [rsp+28h] [rbp-108h]
  unsigned __int64 v19; // [rsp+30h] [rbp-100h]
  int v20; // [rsp+3Ch] [rbp-F4h]
  unsigned int v21; // [rsp+4Ch] [rbp-E4h] BYREF
  _QWORD *v22; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD *v23; // [rsp+58h] [rbp-D8h] BYREF
  _BYTE *v24; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-C8h]
  _BYTE v26[64]; // [rsp+70h] [rbp-C0h] BYREF
  char *v27; // [rsp+B0h] [rbp-80h]
  __int64 v28; // [rsp+B8h] [rbp-78h]
  char v29; // [rsp+C0h] [rbp-70h] BYREF

  v24 = v26;
  v25 = 0x800000000LL;
  sub_13F9CA0(a2, (__int64)&v24);
  v27 = &v29;
  v28 = 0x400000000LL;
  if ( (_DWORD)v25 )
  {
    v2 = 0;
    v3 = a2 + 56;
    v17 = 8LL * (unsigned int)v25;
LABEL_3:
    while ( 1 )
    {
      v4 = v24;
      v19 = sub_157EBA0(*(_QWORD *)&v24[v2]);
      if ( *(_BYTE *)(v19 + 16) != 26 )
        break;
      v5 = **(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(v19 - 24) != v5 && v5 != *(_QWORD *)(v19 - 48) )
      {
        v6 = *(_QWORD *)(v19 + 40);
        if ( v5 != v6 )
        {
          while ( v6 )
          {
            v18 = sub_157F120(v6);
            if ( !v18 )
            {
              if ( !sub_1456E90(a1) )
                break;
              v2 += 8;
              if ( v2 != v17 )
                goto LABEL_3;
              goto LABEL_27;
            }
            v7 = 0;
            v8 = sub_157EBA0(v18);
            v20 = sub_15F3BE0(v8);
            if ( v20 )
            {
              do
              {
                v9 = sub_15F3BF0(v8, v7);
                if ( v6 != v9 && sub_1377F70(v3, v9) )
                  goto LABEL_15;
              }
              while ( v20 != ++v7 );
            }
            v6 = v18;
            if ( v18 == **(_QWORD **)(a2 + 32) )
              goto LABEL_14;
          }
LABEL_15:
          v4 = v24;
          v11 = 0;
          goto LABEL_16;
        }
      }
LABEL_14:
      v10 = *(_QWORD *)(v19 - 72);
      if ( *(_BYTE *)(v10 + 16) != 75 )
        goto LABEL_15;
      if ( sub_1377F70(v3, *(_QWORD *)(v19 - 48)) )
      {
        v21 = sub_15FF0F0(*(_WORD *)(v10 + 18) & 0x7FFF);
      }
      else
      {
        v13 = *(unsigned __int16 *)(v10 + 18);
        BYTE1(v13) &= ~0x80u;
        v21 = v13;
      }
      v22 = (_QWORD *)sub_146F1B0(a1, *(_QWORD *)(v10 - 48));
      v23 = (_QWORD *)sub_146F1B0(a1, *(_QWORD *)(v10 - 24));
      v22 = (_QWORD *)sub_1472270(a1, (__int64)v22, (_QWORD *)a2);
      v23 = (_QWORD *)sub_1472270(a1, (__int64)v23, (_QWORD *)a2);
      if ( sub_146CEE0(a1, (__int64)v22, a2) && !sub_146CEE0(a1, (__int64)v23, a2) )
      {
        v15 = v22;
        v22 = v23;
        v23 = v15;
        v21 = sub_15FF5D0(v21);
      }
      sub_147DF40(a1, &v21, (__int64 *)&v22, (__int64 *)&v23, 0, v14);
      if ( *((_WORD *)v22 + 12) != 7 || a2 != v22[6] || *(_WORD *)(sub_13A5BC0(v22, a1) + 24) )
        goto LABEL_15;
      v2 += 8;
      if ( v2 == v17 )
        goto LABEL_27;
    }
    v11 = 0;
  }
  else
  {
LABEL_27:
    v4 = v24;
    v11 = 1;
  }
LABEL_16:
  if ( v4 != v26 )
    _libc_free((unsigned __int64)v4);
  return v11;
}
