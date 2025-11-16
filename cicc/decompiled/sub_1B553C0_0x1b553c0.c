// Function: sub_1B553C0
// Address: 0x1b553c0
//
__int64 __fastcall sub_1B553C0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r12
  unsigned __int64 v8; // rbx
  int v9; // r12d
  unsigned int v10; // r15d
  int v11; // r11d
  __int64 *v12; // r10
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // ecx
  _QWORD *v18; // rdi
  int v19; // edx
  _QWORD *v20; // rax
  unsigned int v21; // r13d
  __int64 *v22; // r9
  int v23; // r11d
  unsigned int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rdi
  int v31; // r11d
  __int64 *v32; // r9
  __int64 v33[2]; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD *v34; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v35; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-98h]
  __int64 v37; // [rsp+30h] [rbp-90h]
  __int64 v38; // [rsp+38h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-80h] BYREF
  __int64 v40; // [rsp+48h] [rbp-78h]
  _QWORD *v41; // [rsp+50h] [rbp-70h]
  __int64 v42; // [rsp+58h] [rbp-68h]
  __int64 v43; // [rsp+60h] [rbp-60h]
  unsigned __int64 v44; // [rsp+68h] [rbp-58h]
  _QWORD *v45; // [rsp+70h] [rbp-50h]
  __int64 v46; // [rsp+78h] [rbp-48h]
  __int64 v47; // [rsp+80h] [rbp-40h]
  __int64 *v48; // [rsp+88h] [rbp-38h]

  v33[0] = a2;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 8;
  v39 = sub_22077B0(64);
  v3 = (__int64 *)(v39 + 24);
  v4 = sub_22077B0(512);
  v44 = v39 + 24;
  *(_QWORD *)(v39 + 24) = v4;
  v42 = v4;
  v43 = v4 + 512;
  v48 = v3;
  v46 = v4;
  v47 = v4 + 512;
  v41 = (_QWORD *)v4;
  v45 = (_QWORD *)v4;
  ++v35;
  sub_13B3D40((__int64)&v35, 0);
  sub_1898220((__int64)&v35, v33, &v34);
  LODWORD(v37) = v37 + 1;
  if ( *v34 != -8 )
    --HIDWORD(v37);
  *v34 = v33[0];
  v5 = v45;
  if ( v45 == (_QWORD *)(v47 - 8) )
  {
    sub_1B4ECC0(&v39, v33);
    v6 = v45;
  }
  else
  {
    if ( v45 )
    {
      *v45 = v33[0];
      v5 = v45;
    }
    v6 = v5 + 1;
    v45 = v5 + 1;
  }
  if ( v41 == v6 )
  {
LABEL_26:
    v21 = 1;
    goto LABEL_45;
  }
LABEL_7:
  if ( v6 == (_QWORD *)v46 )
  {
    v7 = *(_QWORD *)(*(v48 - 1) + 504);
    j_j___libc_free_0(v6, 512);
    v25 = *--v48 + 512;
    v46 = *v48;
    v47 = v25;
    v45 = (_QWORD *)(v46 + 504);
  }
  else
  {
    v7 = *(v6 - 1);
    v45 = v6 - 1;
  }
  v8 = sub_157EBA0(v7);
  v9 = sub_15F4D60(v8);
  if ( v9 )
  {
    v10 = 0;
    while ( 1 )
    {
      v16 = sub_15F4DF0(v8, v10);
      v34 = (_QWORD *)v16;
      if ( v16 == a1 )
        goto LABEL_12;
      if ( !(_DWORD)v38 )
      {
        ++v35;
LABEL_16:
        sub_13B3D40((__int64)&v35, 2 * v38);
        if ( !(_DWORD)v38 )
          goto LABEL_71;
        v16 = (__int64)v34;
        v17 = (v38 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v12 = (__int64 *)(v36 + 8LL * v17);
        v18 = (_QWORD *)*v12;
        v19 = v37 + 1;
        if ( (_QWORD *)*v12 != v34 )
        {
          v31 = 1;
          v32 = 0;
          while ( v18 != (_QWORD *)-8LL )
          {
            if ( !v32 && v18 == (_QWORD *)-16LL )
              v32 = v12;
            v17 = (v38 - 1) & (v31 + v17);
            v12 = (__int64 *)(v36 + 8LL * v17);
            v18 = (_QWORD *)*v12;
            if ( v34 == (_QWORD *)*v12 )
              goto LABEL_18;
            ++v31;
          }
          if ( v32 )
            v12 = v32;
        }
        goto LABEL_18;
      }
      v11 = 1;
      v12 = 0;
      v13 = (v38 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = (__int64 *)(v36 + 8LL * v13);
      v15 = *v14;
      if ( v16 == *v14 )
      {
LABEL_12:
        if ( v9 == ++v10 )
          goto LABEL_25;
      }
      else
      {
        while ( v15 != -8 )
        {
          if ( v12 || v15 != -16 )
            v14 = v12;
          v13 = (v38 - 1) & (v11 + v13);
          v15 = *(_QWORD *)(v36 + 8LL * v13);
          if ( v16 == v15 )
            goto LABEL_12;
          ++v11;
          v12 = v14;
          v14 = (__int64 *)(v36 + 8LL * v13);
        }
        if ( !v12 )
          v12 = v14;
        ++v35;
        v19 = v37 + 1;
        if ( 4 * ((int)v37 + 1) >= (unsigned int)(3 * v38) )
          goto LABEL_16;
        if ( (int)v38 - HIDWORD(v37) - v19 <= (unsigned int)v38 >> 3 )
        {
          sub_13B3D40((__int64)&v35, v38);
          if ( !(_DWORD)v38 )
          {
LABEL_71:
            LODWORD(v37) = v37 + 1;
            BUG();
          }
          v22 = 0;
          v23 = 1;
          v19 = v37 + 1;
          v24 = (v38 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
          v12 = (__int64 *)(v36 + 8LL * v24);
          v16 = *v12;
          if ( v34 != (_QWORD *)*v12 )
          {
            while ( v16 != -8 )
            {
              if ( v16 == -16 && !v22 )
                v22 = v12;
              v24 = (v38 - 1) & (v23 + v24);
              v12 = (__int64 *)(v36 + 8LL * v24);
              v16 = *v12;
              if ( v34 == (_QWORD *)*v12 )
                goto LABEL_18;
              ++v23;
            }
            v16 = (__int64)v34;
            if ( v22 )
              v12 = v22;
          }
        }
LABEL_18:
        LODWORD(v37) = v19;
        if ( *v12 != -8 )
          --HIDWORD(v37);
        *v12 = v16;
        if ( (unsigned int)v37 > 0xA )
          break;
        v20 = v45;
        if ( v45 == (_QWORD *)(v47 - 8) )
        {
          sub_1B4ECC0(&v39, &v34);
          goto LABEL_12;
        }
        if ( v45 )
        {
          *v45 = v34;
          v20 = v45;
        }
        ++v10;
        v45 = v20 + 1;
        if ( v9 == v10 )
        {
LABEL_25:
          v6 = v45;
          if ( v45 == v41 )
            goto LABEL_26;
          goto LABEL_7;
        }
      }
    }
  }
  v21 = 0;
LABEL_45:
  v26 = v39;
  if ( v39 )
  {
    v27 = (__int64 *)v44;
    v28 = v48 + 1;
    if ( (unsigned __int64)(v48 + 1) > v44 )
    {
      do
      {
        v29 = *v27++;
        j_j___libc_free_0(v29, 512);
      }
      while ( v28 > v27 );
      v26 = v39;
    }
    j_j___libc_free_0(v26, 8 * v40);
  }
  j___libc_free_0(v36);
  return v21;
}
