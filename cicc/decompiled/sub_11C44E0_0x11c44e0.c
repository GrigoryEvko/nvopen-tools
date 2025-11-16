// Function: sub_11C44E0
// Address: 0x11c44e0
//
char __fastcall sub_11C44E0(__int64 **a1, __int64 a2, unsigned int a3)
{
  __int64 *v3; // rbx
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r15
  int v12; // eax
  __int64 v13; // r14
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  bool v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int128 v25; // [rsp-20h] [rbp-F0h]
  __int128 v26; // [rsp-20h] [rbp-F0h]
  __int64 v27; // [rsp+10h] [rbp-C0h]
  __int64 v28; // [rsp+20h] [rbp-B0h]
  __int64 i; // [rsp+28h] [rbp-A8h]
  unsigned __int8 *v30; // [rsp+30h] [rbp-A0h]
  __int64 v31; // [rsp+38h] [rbp-98h]
  __int64 v32; // [rsp+40h] [rbp-90h]
  __int64 v33; // [rsp+40h] [rbp-90h]
  __int64 v34; // [rsp+48h] [rbp-88h] BYREF
  __int64 v35; // [rsp+50h] [rbp-80h] BYREF
  __int64 v36; // [rsp+58h] [rbp-78h] BYREF
  __int64 v37; // [rsp+60h] [rbp-70h]
  __int64 v38; // [rsp+68h] [rbp-68h]
  unsigned __int8 *v39; // [rsp+70h] [rbp-60h]
  __int64 v40[10]; // [rsp+80h] [rbp-50h] BYREF

  v34 = a2;
  if ( a3 )
  {
    v32 = 0;
    v27 = a3;
    do
    {
      a2 = (unsigned int)v32;
      v36 = sub_A744E0(&v34, v32);
      v3 = (__int64 *)sub_A73280(&v36);
      for ( i = sub_A73290(&v36); (__int64 *)i != v3; ++v3 )
      {
        while ( 1 )
        {
          v35 = *v3;
          if ( !sub_A71B30(&v35, 43) )
          {
            a2 = 86;
            if ( !sub_A71B30(&v35, 86) )
              break;
          }
          a2 = (unsigned int)v32;
          v7 = **a1;
          if ( (unsigned __int8)sub_B49B80(v7, v32, 40) )
            break;
          a2 = (unsigned int)v32;
          if ( (unsigned __int8)sub_B49B80(v7, v32, 90) )
            break;
          a2 = (unsigned int)v32;
          if ( (unsigned __int8)sub_B49B80(v7, v32, 91) )
            break;
          if ( (__int64 *)i == ++v3 )
            goto LABEL_18;
        }
        v31 = (__int64)a1[1];
        v30 = *(unsigned __int8 **)(**a1 + 32 * (v32 - (*(_DWORD *)(**a1 + 4) & 0x7FFFFFF)));
        v40[0] = v35;
        if ( !sub_A71860((__int64)v40) && !sub_A71840((__int64)v40) )
        {
          if ( LOBYTE(qword_4F90DA8[8])
            || (v4 = sub_A71AE0(v40), v4 == 5)
            || (v5 = (unsigned int)(v4 - 40), (unsigned int)v5 <= 0x33) && (v6 = 0xC400000000009LL, _bittest64(&v6, v5)) )
          {
            v19 = sub_A71820((__int64)v40);
            v20 = 0;
            if ( v19 )
              v20 = sub_A71B80(v40);
            v28 = v20;
            LODWORD(v37) = sub_A71AE0(v40);
            v38 = v28;
            v39 = v30;
            *((_QWORD *)&v26 + 1) = v28;
            *(_QWORD *)&v26 = v37;
            sub_11C1FA0(v31, a2, v28, v21, v22, v23, v26, v30);
          }
        }
      }
LABEL_18:
      ++v32;
    }
    while ( v27 != v32 );
  }
  v35 = sub_A74680(&v34);
  v8 = sub_A73280(&v35);
  v9 = sub_A73290(&v35);
  v10 = (__int64 *)v9;
  if ( v8 != v9 )
  {
    v11 = (__int64 *)v8;
    do
    {
      while ( 1 )
      {
        v13 = (__int64)a1[1];
        v36 = *v11;
        LOBYTE(v9) = sub_A71860((__int64)&v36);
        if ( !(_BYTE)v9 )
        {
          LOBYTE(v9) = sub_A71840((__int64)&v36);
          if ( !(_BYTE)v9 )
          {
            if ( LOBYTE(qword_4F90DA8[8]) )
              break;
            v12 = sub_A71AE0(&v36);
            if ( v12 == 5 )
              break;
            v9 = (unsigned int)(v12 - 40);
            if ( (unsigned int)v9 <= 0x33 )
            {
              a2 = 0xC400000000009LL;
              if ( _bittest64(&a2, v9) )
                break;
            }
          }
        }
        if ( v10 == ++v11 )
          return v9;
      }
      v14 = sub_A71820((__int64)&v36);
      v15 = 0;
      if ( v14 )
        v15 = sub_A71B80(&v36);
      v33 = v15;
      ++v11;
      LODWORD(v40[0]) = sub_A71AE0(&v36);
      v40[1] = v33;
      v40[2] = 0;
      *((_QWORD *)&v25 + 1) = v33;
      *(_QWORD *)&v25 = v40[0];
      LOBYTE(v9) = sub_11C1FA0(v13, a2, v33, v16, v17, v18, v25, 0);
    }
    while ( v10 != v11 );
  }
  return v9;
}
