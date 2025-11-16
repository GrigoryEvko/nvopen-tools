// Function: sub_272E010
// Address: 0x272e010
//
__int64 __fastcall sub_272E010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  char *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 i; // r13
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  char *v27; // rdi
  __int64 v28; // r15
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  char *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // [rsp+8h] [rbp-108h]
  signed __int64 v35; // [rsp+10h] [rbp-100h]
  __int64 v36; // [rsp+18h] [rbp-F8h]
  signed __int64 v37; // [rsp+20h] [rbp-F0h]
  __int64 v38; // [rsp+28h] [rbp-E8h]
  char *v39; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v40; // [rsp+38h] [rbp-D8h]
  _BYTE v41[128]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+C0h] [rbp-50h]
  __int64 v43; // [rsp+C8h] [rbp-48h]
  unsigned int v44; // [rsp+D0h] [rbp-40h]

  result = a3;
  v36 = a1;
  if ( a1 != a2 )
  {
    result = a1;
    if ( a2 != a3 )
    {
      v8 = 0xCF3CF3CF3CF3CF3DLL;
      v9 = a1 + a3 - a2;
      v34 = v9;
      v35 = 0xCF3CF3CF3CF3CF3DLL * ((a3 - a1) >> 3);
      v38 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - a1) >> 3);
      if ( v38 != v35 - v38 )
      {
        while ( 1 )
        {
          v37 = v35 - v38;
          if ( v38 >= v35 - v38 )
          {
            v19 = v36 + 168 * v35;
            v20 = v19 - 168 * v37;
            v36 = v20;
            if ( v38 > 0 )
            {
              v21 = v20 - 168;
              for ( i = 0; i != v38; ++i )
              {
                v39 = v41;
                v19 -= 168;
                v40 = 0x800000000LL;
                if ( *(_DWORD *)(v21 + 8) )
                  sub_272D8A0((__int64)&v39, (char **)v21, v20, v8, a5, a6);
                v42 = *(_QWORD *)(v21 + 144);
                v43 = *(_QWORD *)(v21 + 152);
                v44 = *(_DWORD *)(v21 + 160);
                sub_272D8A0(v21, (char **)v19, v20, v44, a5, a6);
                *(_QWORD *)(v21 + 144) = *(_QWORD *)(v19 + 144);
                *(_QWORD *)(v21 + 152) = *(_QWORD *)(v19 + 152);
                v23 = *(unsigned int *)(v19 + 160);
                *(_DWORD *)(v21 + 160) = v23;
                sub_272D8A0(v19, &v39, v24, v23, v25, v26);
                v27 = v39;
                *(_QWORD *)(v19 + 144) = v42;
                *(_QWORD *)(v19 + 152) = v43;
                v8 = v44;
                *(_DWORD *)(v19 + 160) = v44;
                if ( v27 != v41 )
                  _libc_free((unsigned __int64)v27);
                v21 -= 168;
              }
              v36 += -168 * v38;
            }
            v38 = v35 % v37;
            if ( !(v35 % v37) )
              return v34;
          }
          else
          {
            v10 = v36;
            v11 = v36 + 168 * v38;
            if ( v35 - v38 > 0 )
            {
              v12 = 0;
              do
              {
                v18 = *(unsigned int *)(v10 + 8);
                v39 = v41;
                v40 = 0x800000000LL;
                if ( (_DWORD)v18 )
                  sub_272D8A0((__int64)&v39, (char **)v10, v18, v8, a5, a6);
                v42 = *(_QWORD *)(v10 + 144);
                v43 = *(_QWORD *)(v10 + 152);
                v44 = *(_DWORD *)(v10 + 160);
                sub_272D8A0(v10, (char **)v11, v18, v44, a5, a6);
                *(_QWORD *)(v10 + 144) = *(_QWORD *)(v11 + 144);
                *(_QWORD *)(v10 + 152) = *(_QWORD *)(v11 + 152);
                v13 = *(unsigned int *)(v11 + 160);
                *(_DWORD *)(v10 + 160) = v13;
                sub_272D8A0(v11, &v39, v14, v13, v15, v16);
                v17 = v39;
                *(_QWORD *)(v11 + 144) = v42;
                *(_QWORD *)(v11 + 152) = v43;
                v8 = v44;
                *(_DWORD *)(v11 + 160) = v44;
                if ( v17 != v41 )
                  _libc_free((unsigned __int64)v17);
                v10 += 168;
                v11 += 168;
                ++v12;
              }
              while ( v37 != v12 );
              v36 += 168 * v37;
            }
            if ( !(v35 % v38) )
              return v34;
            v37 = v38;
            v38 -= v35 % v38;
          }
          v35 = v37;
        }
      }
      v28 = a2;
      do
      {
        v39 = v41;
        v40 = 0x800000000LL;
        v33 = *(unsigned int *)(v36 + 8);
        if ( (_DWORD)v33 )
          sub_272D8A0((__int64)&v39, (char **)v36, v9, v33, a5, a6);
        v42 = *(_QWORD *)(v36 + 144);
        v43 = *(_QWORD *)(v36 + 152);
        v44 = *(_DWORD *)(v36 + 160);
        sub_272D8A0(v36, (char **)v28, v36, v33, a5, a6);
        *(_QWORD *)(v36 + 144) = *(_QWORD *)(v28 + 144);
        *(_QWORD *)(v36 + 152) = *(_QWORD *)(v28 + 152);
        *(_DWORD *)(v36 + 160) = *(_DWORD *)(v28 + 160);
        sub_272D8A0(v28, &v39, v36, v29, v30, v31);
        v32 = v39;
        *(_QWORD *)(v28 + 144) = v42;
        *(_QWORD *)(v28 + 152) = v43;
        *(_DWORD *)(v28 + 160) = v44;
        if ( v32 != v41 )
          _libc_free((unsigned __int64)v32);
        v36 += 168;
        v28 += 168;
      }
      while ( a2 != v36 );
      return a2;
    }
  }
  return result;
}
