// Function: sub_161BE80
// Address: 0x161be80
//
__int64 __fastcall sub_161BE80(_QWORD *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  const char *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 *v16; // r9
  unsigned __int64 *v17; // r8
  unsigned __int64 *v18; // r9
  unsigned __int64 *v19; // r8
  char *v20; // r15
  __int64 v21; // r10
  unsigned __int64 *v22; // r8
  __int64 *v23; // r13
  __int64 *v24; // r15
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r12
  unsigned __int64 *v32; // rax
  __int64 v33; // r13
  unsigned __int64 *v34; // rdx
  unsigned __int64 *v35; // rcx
  unsigned __int64 v36; // rdx
  unsigned __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r13
  char *v40; // rax
  unsigned __int64 *v41; // [rsp+8h] [rbp-C8h]
  unsigned __int64 *v42; // [rsp+10h] [rbp-C0h]
  unsigned __int64 *v43; // [rsp+10h] [rbp-C0h]
  unsigned __int64 *v44; // [rsp+18h] [rbp-B8h]
  __int64 v45; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v47; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+38h] [rbp-98h]
  _BYTE v49[16]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v50; // [rsp+50h] [rbp-80h] BYREF
  __int64 v51; // [rsp+58h] [rbp-78h]
  _BYTE v52[112]; // [rsp+60h] [rbp-70h] BYREF

  v6 = sub_1643360(*a1);
  v50 = v52;
  v7 = v6;
  v51 = 0x800000000LL;
  if ( a3 )
  {
    v8 = "synthetic_function_entry_count";
    v9 = 30;
  }
  else
  {
    v8 = "function_entry_count";
    v9 = 20;
  }
  *(_QWORD *)&v50[8 * (unsigned int)v51] = sub_161BD10(a1, (__int64)v8, v9);
  LODWORD(v51) = v51 + 1;
  v10 = sub_15A0680(v7, a2, 0);
  v13 = sub_161BD20((__int64)a1, v10, v11, v12);
  v14 = (unsigned int)v51;
  if ( (unsigned int)v51 >= HIDWORD(v51) )
  {
    sub_16CD150(&v50, v52, 0, 8);
    v14 = (unsigned int)v51;
  }
  *(_QWORD *)&v50[8 * v14] = v13;
  v15 = (unsigned int)(v51 + 1);
  LODWORD(v51) = v51 + 1;
  if ( a4 )
  {
    v16 = *(unsigned __int64 **)(a4 + 8);
    v17 = &v16[*(unsigned int *)(a4 + 24)];
    if ( !*(_DWORD *)(a4 + 16) || v16 == v17 )
    {
LABEL_7:
      HIDWORD(v48) = 2;
      v47 = (unsigned __int64 *)v49;
    }
    else
    {
      while ( *v16 > 0xFFFFFFFFFFFFFFFDLL )
      {
        if ( v17 == ++v16 )
          goto LABEL_7;
      }
      v47 = (unsigned __int64 *)v49;
      v48 = 0x200000000LL;
      if ( v17 != v16 )
      {
        v32 = v16;
        v33 = 0;
        while ( 1 )
        {
          v34 = v32 + 1;
          if ( v17 == v32 + 1 )
            break;
          while ( 1 )
          {
            v32 = v34;
            if ( *v34 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            if ( v17 == ++v34 )
              goto LABEL_32;
          }
          ++v33;
          if ( v17 == v34 )
            goto LABEL_33;
        }
LABEL_32:
        ++v33;
LABEL_33:
        v35 = (unsigned __int64 *)v49;
        if ( v33 > 2 )
        {
          v42 = v16;
          v44 = v17;
          sub_16CD150(&v47, v49, v33, 8);
          v17 = v44;
          v16 = v42;
          v35 = &v47[(unsigned int)v48];
        }
        v36 = *v16;
        do
        {
          v37 = v16 + 1;
          *v35++ = v36;
          if ( v17 == v16 + 1 )
            break;
          while ( 1 )
          {
            v36 = *v37;
            v16 = v37;
            if ( *v37 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            if ( v17 == ++v37 )
              goto LABEL_39;
          }
        }
        while ( v17 != v37 );
LABEL_39:
        v19 = v47;
        LODWORD(v48) = v48 + v33;
        v38 = 8LL * (unsigned int)v48;
        v18 = &v47[(unsigned __int64)v38 / 8];
        v39 = v38 >> 3;
        if ( v38 )
        {
          while ( 1 )
          {
            v41 = v19;
            v43 = v18;
            v40 = (char *)sub_2207800(8 * v39, &unk_435FF63);
            v18 = v43;
            v19 = v41;
            v20 = v40;
            if ( v40 )
              break;
            v39 >>= 1;
            if ( !v39 )
              goto LABEL_9;
          }
          sub_161BC40(v41, v43, v40, (char *)v39);
          v21 = 8 * v39;
          goto LABEL_10;
        }
LABEL_9:
        v20 = 0;
        sub_161B760(v19, v18);
        v21 = 0;
LABEL_10:
        j_j___libc_free_0(v20, v21);
        v22 = v47;
        v23 = (__int64 *)&v47[(unsigned int)v48];
        if ( v23 != (__int64 *)v47 )
        {
          v24 = (__int64 *)v47;
          do
          {
            v25 = sub_15A0680(v7, *v24, 0);
            v28 = sub_161BD20((__int64)a1, v25, v26, v27);
            v29 = (unsigned int)v51;
            if ( (unsigned int)v51 >= HIDWORD(v51) )
            {
              v45 = v28;
              sub_16CD150(&v50, v52, 0, 8);
              v29 = (unsigned int)v51;
              v28 = v45;
            }
            ++v24;
            *(_QWORD *)&v50[8 * v29] = v28;
            LODWORD(v51) = v51 + 1;
          }
          while ( v23 != v24 );
          v22 = v47;
        }
        if ( v22 != (unsigned __int64 *)v49 )
          _libc_free((unsigned __int64)v22);
        v15 = (unsigned int)v51;
        goto LABEL_19;
      }
    }
    v18 = (unsigned __int64 *)v49;
    LODWORD(v48) = 0;
    v19 = (unsigned __int64 *)v49;
    goto LABEL_9;
  }
LABEL_19:
  v30 = sub_1627350(*a1, v50, v15, 0, 1);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  return v30;
}
