// Function: sub_2C9E510
// Address: 0x2c9e510
//
__int64 __fastcall sub_2C9E510(_BYTE *a1, unsigned int a2)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi
  unsigned int v4; // r12d
  __int64 *v5; // rcx
  unsigned int v6; // esi
  __int64 v7; // rbx
  int v8; // r11d
  _QWORD *v9; // r10
  unsigned int v10; // edi
  _QWORD *v11; // rdx
  __int64 v12; // rax
  int v14; // eax
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 v17; // rdx
  _BYTE *v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // edx
  __int64 v21; // rdi
  int v22; // r11d
  _QWORD *v23; // r9
  _QWORD *v24; // r8
  unsigned int v25; // r13d
  int v26; // r9d
  __int64 v27; // rsi
  __int64 *v29; // [rsp+18h] [rbp-B8h]
  __int64 *v30; // [rsp+18h] [rbp-B8h]
  __int64 *v31; // [rsp+18h] [rbp-B8h]
  __int64 *v32; // [rsp+18h] [rbp-B8h]
  _BYTE *v33; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v35; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-98h]
  __int64 v37; // [rsp+40h] [rbp-90h]
  __int64 v38; // [rsp+48h] [rbp-88h]
  __int64 v39[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v40; // [rsp+60h] [rbp-70h]
  __int64 v41; // [rsp+68h] [rbp-68h]
  __int64 v42; // [rsp+70h] [rbp-60h]
  __int64 v43; // [rsp+78h] [rbp-58h]
  _QWORD *v44; // [rsp+80h] [rbp-50h]
  _QWORD *v45; // [rsp+88h] [rbp-48h]
  __int64 v46; // [rsp+90h] [rbp-40h]
  _QWORD *v47; // [rsp+98h] [rbp-38h]

  if ( *a1 <= 0x1Cu )
  {
    return 0;
  }
  else
  {
    v33 = a1;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39[0] = 0;
    v39[1] = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    sub_2785050(v39, 0);
    v2 = v44;
    if ( v44 == (_QWORD *)(v46 - 8) )
    {
      sub_2785520((unsigned __int64 *)v39, &v33);
      v3 = v44;
    }
    else
    {
      if ( v44 )
      {
        *v44 = v33;
        v2 = v44;
      }
      v3 = v2 + 1;
      v44 = v3;
    }
    v4 = 0;
    v5 = &v34;
    while ( v40 != v3 )
    {
      while ( 1 )
      {
        if ( v45 == v3 )
        {
          v29 = v5;
          v7 = *(_QWORD *)(*(v47 - 1) + 504LL);
          j_j___libc_free_0((unsigned __int64)v3);
          v6 = v38;
          v5 = v29;
          v19 = *--v47 + 512LL;
          v45 = (_QWORD *)*v47;
          v46 = v19;
          v44 = v45 + 63;
          if ( !(_DWORD)v38 )
          {
LABEL_38:
            ++v35;
            goto LABEL_39;
          }
        }
        else
        {
          v6 = v38;
          v7 = *(v3 - 1);
          v44 = v3 - 1;
          if ( !(_DWORD)v38 )
            goto LABEL_38;
        }
        v8 = 1;
        v9 = 0;
        v10 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v11 = (_QWORD *)(v36 + 8LL * v10);
        v12 = *v11;
        if ( *v11 != v7 )
          break;
LABEL_11:
        v3 = v44;
        if ( v40 == v44 )
          goto LABEL_12;
      }
      while ( v12 != -4096 )
      {
        if ( v9 || v12 != -8192 )
          v11 = v9;
        v10 = (v6 - 1) & (v8 + v10);
        v12 = *(_QWORD *)(v36 + 8LL * v10);
        if ( v12 == v7 )
          goto LABEL_11;
        ++v8;
        v9 = v11;
        v11 = (_QWORD *)(v36 + 8LL * v10);
      }
      if ( !v9 )
        v9 = v11;
      ++v35;
      v14 = v37 + 1;
      if ( 4 * ((int)v37 + 1) >= 3 * v6 )
      {
LABEL_39:
        v30 = v5;
        sub_CF4090((__int64)&v35, 2 * v6);
        if ( !(_DWORD)v38 )
          goto LABEL_66;
        v5 = v30;
        v20 = (v38 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v9 = (_QWORD *)(v36 + 8LL * v20);
        v21 = *v9;
        v14 = v37 + 1;
        if ( *v9 != v7 )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -4096 )
          {
            if ( !v23 && v21 == -8192 )
              v23 = v9;
            v20 = (v38 - 1) & (v22 + v20);
            v9 = (_QWORD *)(v36 + 8LL * v20);
            v21 = *v9;
            if ( *v9 == v7 )
              goto LABEL_23;
            ++v22;
          }
          if ( v23 )
            v9 = v23;
        }
      }
      else if ( v6 - (v14 + HIDWORD(v37)) <= v6 >> 3 )
      {
        v32 = v5;
        sub_CF4090((__int64)&v35, v6);
        if ( !(_DWORD)v38 )
        {
LABEL_66:
          LODWORD(v37) = v37 + 1;
          BUG();
        }
        v24 = 0;
        v5 = v32;
        v25 = (v38 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v26 = 1;
        v9 = (_QWORD *)(v36 + 8LL * v25);
        v27 = *v9;
        v14 = v37 + 1;
        if ( v7 != *v9 )
        {
          while ( v27 != -4096 )
          {
            if ( v27 == -8192 && !v24 )
              v24 = v9;
            v25 = (v38 - 1) & (v26 + v25);
            v9 = (_QWORD *)(v36 + 8LL * v25);
            v27 = *v9;
            if ( *v9 == v7 )
              goto LABEL_23;
            ++v26;
          }
          if ( v24 )
            v9 = v24;
        }
      }
LABEL_23:
      LODWORD(v37) = v14;
      if ( *v9 != -4096 )
        --HIDWORD(v37);
      *v9 = v7;
      if ( ++v4 > a2 )
        break;
      v3 = v44;
      v15 = 0;
      v16 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
      {
        do
        {
          if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
            v17 = *(_QWORD *)(v7 - 8);
          else
            v17 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
          v18 = *(_BYTE **)(v17 + v15);
          if ( *v18 > 0x1Cu )
          {
            v34 = *(_QWORD *)(v17 + v15);
            if ( v3 == (_QWORD *)(v46 - 8) )
            {
              v31 = v5;
              sub_2785520((unsigned __int64 *)v39, v5);
              v3 = v44;
              v5 = v31;
            }
            else
            {
              if ( v3 )
              {
                *v3 = v18;
                v3 = v44;
              }
              v44 = ++v3;
            }
          }
          v15 += 32;
        }
        while ( v16 != v15 );
      }
    }
LABEL_12:
    sub_2784FD0((unsigned __int64 *)v39);
    sub_C7D6A0(v36, 8LL * (unsigned int)v38, 8);
  }
  return v4;
}
