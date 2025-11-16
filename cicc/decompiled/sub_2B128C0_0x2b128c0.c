// Function: sub_2B128C0
// Address: 0x2b128c0
//
void __fastcall sub_2B128C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rdx
  char **v19; // r14
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+18h] [rbp-78h]
  __int64 v31; // [rsp+18h] [rbp-78h]
  char *v32[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v33[96]; // [rsp+30h] [rbp-60h] BYREF

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v16 = a1;
        v17 = a2;
LABEL_12:
        if ( *(_DWORD *)(v17 + 16) > *(_DWORD *)(v16 + 16) )
        {
          v18 = *(_QWORD *)v16;
          v19 = (char **)(v16 + 8);
          *(_QWORD *)v16 = *(_QWORD *)v17;
          v20 = 0xC00000000LL;
          *(_QWORD *)v17 = v18;
          v21 = *(_DWORD *)(v16 + 16);
          v32[0] = v33;
          v32[1] = (char *)0xC00000000LL;
          if ( v21 )
          {
            v31 = v17;
            sub_2B0D090((__int64)v32, v19, v18, 0xC00000000LL, a5, v7);
            v17 = v31;
          }
          v22 = v17 + 8;
          sub_2B0D090((__int64)v19, (char **)(v17 + 8), v18, v20, a5, v7);
          sub_2B0D090(v22, v32, v23, v24, v25, v26);
          if ( v32[0] != v33 )
            _libc_free((unsigned __int64)v32[0]);
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_2B0EA80(v7, a3, v6 + 72 * (v5 / 2));
        v15 = 0x8E38E38E38E38E39LL * ((v13 - v12) >> 3);
        while ( 1 )
        {
          v28 = v13;
          v30 = v14;
          v8 -= v15;
          v29 = sub_2B11FA0(v14, v12, v13, v10, v11, v12);
          sub_2B128C0(v6, v30, v29, v9, v15);
          v5 -= v9;
          if ( !v5 )
            break;
          v16 = v29;
          v17 = v28;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = v29;
          v7 = v28;
          if ( v5 > v8 )
            goto LABEL_5;
LABEL_10:
          v15 = v8 / 2;
          v14 = sub_2B0EAE0(v6, v7, v7 + 72 * (v8 / 2));
          v9 = 0x8E38E38E38E38E39LL * ((v14 - v6) >> 3);
        }
      }
    }
  }
}
