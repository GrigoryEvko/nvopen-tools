// Function: sub_CBF4B0
// Address: 0xcbf4b0
//
unsigned __int64 __fastcall sub_CBF4B0(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r9
  unsigned int *v3; // r10
  unsigned __int64 v4; // rax
  unsigned int *v6; // rsi
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r11
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned int *v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // rax

  v2 = (unsigned __int64)a1 + a2;
  v3 = a1;
  v4 = 0x27D4EB2F165667C5LL;
  if ( a2 > 0x1F )
  {
    v6 = a1;
    v7 = 0;
    v8 = 0xC2B2AE3D27D4EB4FLL;
    v9 = 0x61C8864E7A143579LL;
    v10 = 0x60EA27EEADC0B5D6LL;
    do
    {
      v11 = *(_QWORD *)v6;
      v12 = *((_QWORD *)v6 + 1);
      v6 += 8;
      v13 = __ROL8__(v10 - 0x3D4D51C2D82B14B1LL * v11, 31);
      v14 = __ROL8__(v8 - 0x3D4D51C2D82B14B1LL * v12, 31);
      v15 = __ROL8__(v7 - 0x3D4D51C2D82B14B1LL * *((_QWORD *)v6 - 2), 31);
      v10 = 0x9E3779B185EBCA87LL * v13;
      v16 = __ROL8__(v9 - 0x3D4D51C2D82B14B1LL * *((_QWORD *)v6 - 1), 31);
      v8 = 0x9E3779B185EBCA87LL * v14;
      v7 = 0x9E3779B185EBCA87LL * v15;
      v9 = 0x9E3779B185EBCA87LL * v16;
    }
    while ( v2 - 32 >= (unsigned __int64)v6 );
    v17 = ((a2 - 32) & 0xFFFFFFFFFFFFFFE0LL) + 32;
    if ( v2 - 31 < (unsigned __int64)a1 + 1 )
      v17 = 32;
    v3 = (unsigned int *)((char *)a1 + v17);
    v4 = 0x9E3779B185EBCA87LL
       * ((0x9E3779B185EBCA87LL
         * ((0x9E3779B185EBCA87LL
           * ((0x9E3779B185EBCA87LL * __ROL8__(0xDEF35B010F796CA9LL * v14, 31))
            ^ (0x9E3779B185EBCA87LL
             * ((0x9E3779B185EBCA87LL * __ROL8__(0xDEF35B010F796CA9LL * v13, 31))
              ^ (__ROL8__(v9, 18) + __ROL8__(v7, 12) + __ROL8__(v8, 7) + __ROL8__(v10, 1)))
             - 0x7A1435883D4D519DLL))
           - 0x7A1435883D4D519DLL)
          ^ (0x9E3779B185EBCA87LL * __ROL8__(0xDEF35B010F796CA9LL * v15, 31)))
         - 0x7A1435883D4D519DLL)
        ^ (0x9E3779B185EBCA87LL * __ROL8__(0xDEF35B010F796CA9LL * v16, 31)))
       - 0x7A1435883D4D519DLL;
  }
  v18 = v3 + 2;
  v19 = a2 + v4;
  if ( v2 >= (unsigned __int64)(v3 + 2) )
  {
    v20 = v19;
    do
    {
      v21 = *(_QWORD *)v3;
      v3 = v18;
      v18 += 2;
      v22 = 0x9E3779B185EBCA87LL * __ROL8__(v20 ^ (0x9E3779B185EBCA87LL * __ROL8__(0xC2B2AE3D27D4EB4FLL * v21, 31)), 27);
      v20 = v22 - 0x7A1435883D4D519DLL;
    }
    while ( v2 >= (unsigned __int64)v18 );
    v19 = v22 - 0x7A1435883D4D519DLL;
  }
  if ( (unsigned __int64)(v3 + 1) <= v2 )
  {
    v23 = *v3++;
    v19 = 0xC2B2AE3D27D4EB4FLL * __ROL8__((0x9E3779B185EBCA87LL * v23) ^ v19, 23) + 0x165667B19E3779F9LL;
  }
  if ( v2 > (unsigned __int64)v3 )
  {
    v24 = v19;
    do
    {
      v25 = *(unsigned __int8 *)v3;
      v3 = (unsigned int *)((char *)v3 + 1);
      v19 = 0x9E3779B185EBCA87LL * __ROL8__(v24 ^ (0x27D4EB2F165667C5LL * v25), 11);
      v24 = v19;
    }
    while ( (unsigned int *)v2 != v3 );
  }
  v26 = 0x165667B19E3779F9LL
      * (((0xC2B2AE3D27D4EB4FLL * ((v19 >> 33) ^ v19)) >> 29) ^ (0xC2B2AE3D27D4EB4FLL * ((v19 >> 33) ^ v19)));
  return HIDWORD(v26) ^ v26;
}
