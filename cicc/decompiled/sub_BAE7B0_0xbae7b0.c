// Function: sub_BAE7B0
// Address: 0xbae7b0
//
__int64 __fastcall sub_BAE7B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdi
  __int64 v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  __int64 *v25; // r15
  __int64 *v26; // r12
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // rdi
  _QWORD v32[2]; // [rsp+0h] [rbp-140h] BYREF
  _QWORD v33[2]; // [rsp+10h] [rbp-130h] BYREF
  __int64 v34; // [rsp+20h] [rbp-120h]
  __int64 v35[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v36; // [rsp+40h] [rbp-100h]
  __int64 v37[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v38; // [rsp+60h] [rbp-E0h]
  __int64 v39; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-C8h]
  __int64 v41; // [rsp+80h] [rbp-C0h]
  __int64 v42; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+98h] [rbp-A8h]
  __int64 v44; // [rsp+A0h] [rbp-A0h]
  __int64 v45; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v46; // [rsp+B8h] [rbp-88h]
  __int64 v47; // [rsp+C0h] [rbp-80h]
  __int64 v48; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v49; // [rsp+D8h] [rbp-68h]
  __int64 v50; // [rsp+E0h] [rbp-60h]
  _QWORD *v51; // [rsp+F0h] [rbp-50h] BYREF
  _QWORD *v52; // [rsp+F8h] [rbp-48h]
  __int64 v53; // [rsp+100h] [rbp-40h]

  v3 = 193;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v37[0] = 0;
  v37[1] = 0;
  v38 = 0;
  v35[0] = 0;
  v35[1] = 0;
  v36 = 0;
  v33[0] = 0;
  v33[1] = 0;
  v34 = 0;
  v32[0] = v33;
  v32[1] = 0;
  sub_9C6E00(a1, 193, 0, 0, (__int64)v32, a2, v33, v35, v37, &v39, &v42, &v45, &v48, (__int64)&v51);
  if ( (_QWORD *)v32[0] != v33 )
    _libc_free(v32[0], 193);
  if ( v33[0] )
  {
    v3 = v34 - v33[0];
    j_j___libc_free_0(v33[0], v34 - v33[0]);
  }
  if ( v35[0] )
  {
    v3 = v36 - v35[0];
    j_j___libc_free_0(v35[0], v36 - v35[0]);
  }
  if ( v37[0] )
  {
    v3 = v38 - v37[0];
    j_j___libc_free_0(v37[0], v38 - v37[0]);
  }
  v5 = v40;
  v6 = v39;
  if ( v40 != v39 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 + 16);
      if ( v7 )
      {
        v3 = *(_QWORD *)(v6 + 32) - v7;
        j_j___libc_free_0(v7, v3);
      }
      v6 += 40;
    }
    while ( v5 != v6 );
    v6 = v39;
  }
  if ( v6 )
  {
    v3 = v41 - v6;
    j_j___libc_free_0(v6, v41 - v6);
  }
  v8 = v43;
  v9 = v42;
  if ( v43 != v42 )
  {
    do
    {
      v10 = *(_QWORD *)(v9 + 16);
      if ( v10 )
      {
        v3 = *(_QWORD *)(v9 + 32) - v10;
        j_j___libc_free_0(v10, v3);
      }
      v9 += 40;
    }
    while ( v8 != v9 );
    v9 = v42;
  }
  if ( v9 )
  {
    v3 = v44 - v9;
    j_j___libc_free_0(v9, v44 - v9);
  }
  v11 = v46;
  v12 = v45;
  if ( v46 != v45 )
  {
    do
    {
      v13 = *(_QWORD *)(v12 + 48);
      v14 = *(_QWORD *)(v12 + 40);
      if ( v13 != v14 )
      {
        do
        {
          if ( *(_DWORD *)(v14 + 40) > 0x40u )
          {
            v15 = *(_QWORD *)(v14 + 32);
            if ( v15 )
              j_j___libc_free_0_0(v15);
          }
          if ( *(_DWORD *)(v14 + 24) > 0x40u )
          {
            v16 = *(_QWORD *)(v14 + 16);
            if ( v16 )
              j_j___libc_free_0_0(v16);
          }
          v14 += 48;
        }
        while ( v13 != v14 );
        v14 = *(_QWORD *)(v12 + 40);
      }
      if ( v14 )
      {
        v3 = *(_QWORD *)(v12 + 56) - v14;
        j_j___libc_free_0(v14, v3);
      }
      if ( *(_DWORD *)(v12 + 32) > 0x40u )
      {
        v17 = *(_QWORD *)(v12 + 24);
        if ( v17 )
          j_j___libc_free_0_0(v17);
      }
      if ( *(_DWORD *)(v12 + 16) > 0x40u )
      {
        v18 = *(_QWORD *)(v12 + 8);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      v12 += 64;
    }
    while ( v11 != v12 );
    v12 = v45;
  }
  if ( v12 )
  {
    v3 = v47 - v12;
    j_j___libc_free_0(v12, v47 - v12);
  }
  v19 = v49;
  v20 = v48;
  if ( v49 != v48 )
  {
    do
    {
      v21 = *(_QWORD *)(v20 + 72);
      if ( v21 != v20 + 88 )
        _libc_free(v21, v3);
      v22 = *(_QWORD *)(v20 + 8);
      if ( v22 != v20 + 24 )
        _libc_free(v22, v3);
      v20 += 136;
    }
    while ( v19 != v20 );
    v20 = v48;
  }
  if ( v20 )
  {
    v3 = v50 - v20;
    j_j___libc_free_0(v20, v50 - v20);
  }
  v23 = v52;
  v24 = v51;
  if ( v52 != v51 )
  {
    do
    {
      v25 = (__int64 *)v24[12];
      v26 = (__int64 *)v24[11];
      if ( v25 != v26 )
      {
        do
        {
          v27 = *v26;
          if ( *v26 )
          {
            v3 = v26[2] - v27;
            j_j___libc_free_0(v27, v3);
          }
          v26 += 3;
        }
        while ( v25 != v26 );
        v26 = (__int64 *)v24[11];
      }
      if ( v26 )
      {
        v3 = v24[13] - (_QWORD)v26;
        j_j___libc_free_0(v26, v3);
      }
      v28 = v24[9];
      v29 = v24[8];
      if ( v28 != v29 )
      {
        do
        {
          v30 = *(_QWORD *)(v29 + 8);
          if ( v30 != v29 + 24 )
            _libc_free(v30, v3);
          v29 += 72;
        }
        while ( v28 != v29 );
        v29 = v24[8];
      }
      if ( v29 )
      {
        v3 = v24[10] - v29;
        j_j___libc_free_0(v29, v3);
      }
      if ( (_QWORD *)*v24 != v24 + 3 )
        _libc_free(*v24, v3);
      v24 += 14;
    }
    while ( v23 != v24 );
    v24 = v51;
  }
  if ( v24 )
    j_j___libc_free_0(v24, v53 - (_QWORD)v24);
  return a1;
}
