// Function: sub_192F030
// Address: 0x192f030
//
void __fastcall sub_192F030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rcx
  int v11; // r8d
  __int64 v12; // r9
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  _DWORD *v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  int v34; // [rsp+20h] [rbp-80h]
  int v35; // [rsp+24h] [rbp-7Ch]
  int v36; // [rsp+28h] [rbp-78h]
  int v37; // [rsp+2Ch] [rbp-74h]
  int v38; // [rsp+30h] [rbp-70h]
  char *v39[2]; // [rsp+38h] [rbp-68h] BYREF
  _BYTE v40[88]; // [rsp+48h] [rbp-58h] BYREF

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
          v18 = v16 + 24;
          v34 = *(_DWORD *)v16;
          v35 = *(_DWORD *)(v16 + 4);
          v36 = *(_DWORD *)(v16 + 8);
          v38 = *(_DWORD *)(v16 + 16);
          v37 = *(_DWORD *)(v16 + 12);
          v19 = 0x400000000LL;
          v39[1] = (char *)0x400000000LL;
          v20 = *(unsigned int *)(v16 + 32);
          v39[0] = v40;
          if ( (_DWORD)v20 )
          {
            v30 = v16;
            v33 = v17;
            sub_192DBD0((__int64)v39, (char **)(v16 + 24), v20, 0x400000000LL, a5, v7);
            v16 = v30;
            v17 = v33;
          }
          v21 = v17 + 24;
          v32 = (_DWORD *)v17;
          *(_DWORD *)v16 = *(_DWORD *)v17;
          *(_DWORD *)(v16 + 4) = *(_DWORD *)(v17 + 4);
          *(_DWORD *)(v16 + 8) = *(_DWORD *)(v17 + 8);
          *(_DWORD *)(v16 + 12) = *(_DWORD *)(v17 + 12);
          v22 = *(unsigned int *)(v17 + 16);
          *(_DWORD *)(v16 + 16) = v22;
          sub_192DBD0(v18, (char **)(v17 + 24), v22, v19, a5, v7);
          *v32 = v34;
          v32[1] = v35;
          v32[2] = v36;
          v32[3] = v37;
          v32[4] = v38;
          sub_192DBD0(v21, v39, v23, v24, v25, v26);
          if ( v39[0] != v40 )
            _libc_free((unsigned __int64)v39[0]);
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_192E3D0(v7, a3, v6 + 72 * (v5 / 2));
        v15 = 0x8E38E38E38E38E39LL * ((v13 - v12) >> 3);
        while ( 1 )
        {
          v28 = v13;
          v31 = v14;
          v8 -= v15;
          v29 = sub_192EB50(v14, v12, v13, v10, v11, v12);
          sub_192F030(v6, v31, v29, v9, v15);
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
          v14 = sub_192E430(v6, v7, v7 + 72 * (v8 / 2));
          v9 = 0x8E38E38E38E38E39LL * ((v14 - v6) >> 3);
        }
      }
    }
  }
}
