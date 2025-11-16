// Function: sub_1AD3E20
// Address: 0x1ad3e20
//
void __fastcall sub_1AD3E20(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  _QWORD *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rbx
  unsigned __int64 v15; // r8
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // r9
  __int64 *v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 *v24; // rsi
  __int64 *v25; // rcx
  unsigned __int64 v26; // rax
  int v27; // edx
  int v28; // r9d
  __int64 v32; // [rsp+20h] [rbp-110h]
  unsigned __int64 v33; // [rsp+20h] [rbp-110h]
  unsigned __int64 v35[2]; // [rsp+30h] [rbp-100h] BYREF
  __int64 v36; // [rsp+40h] [rbp-F0h]
  __int64 v37; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v38; // [rsp+58h] [rbp-D8h]
  __int64 *v39; // [rsp+60h] [rbp-D0h]
  __int64 v40; // [rsp+68h] [rbp-C8h]
  int v41; // [rsp+70h] [rbp-C0h]
  _BYTE v42[184]; // [rsp+78h] [rbp-B8h] BYREF

  v38 = (__int64 *)v42;
  v39 = (__int64 *)v42;
  v6 = *(_DWORD *)(a2 + 16);
  v37 = 0;
  v40 = 16;
  v41 = 0;
  if ( v6 )
  {
    v14 = *(_QWORD *)(a2 + 8);
    v15 = (unsigned __int64)*(unsigned int *)(a2 + 24) << 6;
    v16 = v14 + v15;
    if ( v14 + v15 != v14 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(v14 + 24);
        if ( v17 != -16 && v17 != -8 )
          break;
        v14 += 64;
        if ( v16 == v14 )
          goto LABEL_2;
      }
      if ( v16 != v14 )
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v14 + 24);
          if ( *(_BYTE *)(v18 + 16) == 18 )
          {
            v19 = *(_QWORD *)(v14 + 56);
            if ( v19 )
              break;
          }
LABEL_25:
          v14 += 64;
          if ( v14 != v16 )
          {
            while ( 1 )
            {
              v23 = *(_QWORD *)(v14 + 24);
              if ( v23 != -16 && v23 != -8 )
                break;
              v14 += 64;
              if ( v16 == v14 )
                goto LABEL_2;
            }
            if ( v16 != v14 )
              continue;
          }
          goto LABEL_2;
        }
        v20 = sub_1368AA0(a4, v18);
        v21 = v38;
        if ( v39 == v38 )
        {
          v24 = &v38[HIDWORD(v40)];
          if ( v38 != v24 )
          {
            v25 = 0;
            while ( v19 != *v21 )
            {
              if ( *v21 == -2 )
                v25 = v21;
              if ( v24 == ++v21 )
              {
                if ( !v25 )
                  goto LABEL_43;
                *v25 = v19;
                --v41;
                ++v37;
                goto LABEL_24;
              }
            }
            goto LABEL_38;
          }
LABEL_43:
          if ( HIDWORD(v40) < (unsigned int)v40 )
          {
            ++HIDWORD(v40);
            *v24 = v19;
            ++v37;
            goto LABEL_24;
          }
        }
        v32 = v20;
        sub_16CCBA0((__int64)&v37, v19);
        v20 = v32;
        if ( v22 )
        {
LABEL_24:
          sub_136C010(a3, v19, v20);
          goto LABEL_25;
        }
LABEL_38:
        v33 = v20;
        v26 = sub_1368AA0(a3, v19);
        v20 = v33;
        if ( v33 < v26 )
          v20 = v26;
        goto LABEL_24;
      }
    }
  }
LABEL_2:
  v7 = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)v7 )
    goto LABEL_3;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = (v7 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
  v13 = v12[3];
  if ( a5 != v13 )
  {
    v27 = 1;
    while ( v13 != -8 )
    {
      v28 = v27 + 1;
      v11 = (v7 - 1) & (v27 + v11);
      v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
      v13 = v12[3];
      if ( a5 == v13 )
        goto LABEL_8;
      v27 = v28;
    }
    goto LABEL_3;
  }
LABEL_8:
  if ( v12 == (_QWORD *)(v10 + (v7 << 6)) )
  {
LABEL_3:
    v8 = 0;
    goto LABEL_4;
  }
  v35[0] = 6;
  v8 = v12[7];
  v35[1] = 0;
  v36 = v8;
  if ( v8 != 0 && v8 != -8 && v8 != -16 )
  {
    sub_1649AC0(v35, v12[5] & 0xFFFFFFFFFFFFFFF8LL);
    v8 = v36;
    if ( v36 != -16 && v36 != 0 && v36 != -8 )
      sub_1649B30(v35);
  }
LABEL_4:
  v9 = sub_1368AA0(a3, a1);
  sub_136C020(a3, v8, v9, (__int64)&v37);
  if ( v39 != v38 )
    _libc_free((unsigned __int64)v39);
}
