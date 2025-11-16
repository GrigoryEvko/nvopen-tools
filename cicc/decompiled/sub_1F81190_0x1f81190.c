// Function: sub_1F81190
// Address: 0x1f81190
//
__int64 __fastcall sub_1F81190(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v6; // rax
  _QWORD *v7; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rdi
  __int64 *v11; // r8
  __int64 v12; // r15
  __int64 *v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // rbx
  __int64 v16; // r10
  __int64 v17; // r14
  char v18; // dl
  __int64 v19; // r15
  __int64 *v20; // rsi
  __int64 *v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // r8
  __int64 *v24; // rbx
  __int64 *v25; // r12
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // r13
  unsigned int v31; // r12d
  __int64 *v33; // rdx
  __int64 *v34; // [rsp+0h] [rbp-1D0h]
  unsigned int v35; // [rsp+0h] [rbp-1D0h]
  __int64 *v37; // [rsp+18h] [rbp-1B8h]
  _QWORD *v38; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v39; // [rsp+28h] [rbp-1A8h]
  _QWORD v40[8]; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v41; // [rsp+70h] [rbp-160h] BYREF
  __int64 *v42; // [rsp+78h] [rbp-158h]
  __int64 *v43; // [rsp+80h] [rbp-150h]
  __int64 v44; // [rsp+88h] [rbp-148h]
  int v45; // [rsp+90h] [rbp-140h]
  _BYTE v46[312]; // [rsp+98h] [rbp-138h] BYREF

  v6 = (__int64 *)v46;
  v7 = v40;
  v40[0] = a3;
  v9 = 1;
  v39 = 0x800000001LL;
  v10 = (__int64 *)v46;
  v41 = 0;
  v42 = (__int64 *)v46;
  v43 = (__int64 *)v46;
  v44 = 32;
  v45 = 0;
  v38 = v40;
  v34 = a1;
  while ( 1 )
  {
    v12 = v7[v9 - 1];
    LODWORD(v39) = v9 - 1;
    if ( *(_WORD *)(v12 + 24) == 2 )
    {
      v23 = *(__int64 **)(v12 + 32);
      v37 = &v23[5 * *(unsigned int *)(v12 + 56)];
      if ( v23 != v37 )
      {
        v24 = *(__int64 **)(v12 + 32);
        while ( 1 )
        {
          v26 = *v24;
          if ( v6 == v10 )
            break;
          v25 = &v10[(unsigned int)v44];
          v6 = sub_16CC9F0((__int64)&v41, *v24);
          if ( v26 == *v6 )
          {
            if ( v43 == v42 )
              v33 = &v43[HIDWORD(v44)];
            else
              v33 = &v43[(unsigned int)v44];
            goto LABEL_45;
          }
          if ( v43 == v42 )
          {
            v6 = &v43[HIDWORD(v44)];
            v33 = v6;
            goto LABEL_45;
          }
          v6 = &v43[(unsigned int)v44];
LABEL_37:
          if ( v6 == v25 )
          {
LABEL_49:
            v27 = (unsigned int)v39;
            if ( (unsigned int)v39 >= HIDWORD(v39) )
            {
              sub_16CD150((__int64)&v38, v40, 0, 8, (int)v23, a6);
              v27 = (unsigned int)v39;
            }
            v38[v27] = v26;
            LODWORD(v39) = v39 + 1;
          }
LABEL_38:
          v10 = v43;
          v6 = v42;
          v24 += 5;
          if ( v37 == v24 )
            goto LABEL_6;
        }
        v25 = &v6[HIDWORD(v44)];
        if ( v25 == v6 )
        {
          v33 = v6;
        }
        else
        {
          do
          {
            if ( v26 == *v6 )
              break;
            ++v6;
          }
          while ( v25 != v6 );
          v33 = v25;
        }
LABEL_45:
        if ( v6 != v33 )
        {
          while ( (unsigned __int64)*v6 >= 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v33 == ++v6 )
            {
              if ( v6 != v25 )
                goto LABEL_38;
              goto LABEL_49;
            }
          }
        }
        goto LABEL_37;
      }
    }
LABEL_6:
    if ( v6 != v10 )
    {
LABEL_2:
      sub_16CCBA0((__int64)&v41, v12);
      v10 = v43;
      v6 = v42;
      goto LABEL_3;
    }
    v11 = &v6[HIDWORD(v44)];
    a6 = HIDWORD(v44);
    if ( v11 == v6 )
    {
LABEL_63:
      if ( HIDWORD(v44) >= (unsigned int)v44 )
        goto LABEL_2;
      a6 = ++HIDWORD(v44);
      *v11 = v12;
      v6 = v42;
      ++v41;
      v10 = v43;
    }
    else
    {
      v13 = v6;
      v14 = 0;
      while ( v12 != *v13 )
      {
        if ( *v13 == -2 )
          v14 = v13;
        if ( v11 == ++v13 )
        {
          if ( !v14 )
            goto LABEL_63;
          *v14 = v12;
          v9 = v39;
          --v45;
          v10 = v43;
          ++v41;
          v6 = v42;
          if ( (_DWORD)v39 )
            goto LABEL_4;
          goto LABEL_15;
        }
      }
    }
LABEL_3:
    v9 = v39;
    if ( !(_DWORD)v39 )
      break;
LABEL_4:
    v7 = v38;
  }
LABEL_15:
  v15 = v34;
  if ( !a2 )
  {
LABEL_65:
    v31 = 1;
    goto LABEL_66;
  }
  v16 = *v34;
  v35 = HIDWORD(v44) + 1024 - v45;
  v17 = 0;
  while ( 2 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16 + v17) + 32LL) + 40LL);
    if ( v10 != v6 )
      goto LABEL_17;
    v20 = &v6[HIDWORD(v44)];
    if ( v20 == v6 )
    {
LABEL_60:
      if ( HIDWORD(v44) < (unsigned int)v44 )
      {
        ++HIDWORD(v44);
        *v20 = v19;
        v22 = (unsigned int)v39;
        ++v41;
        if ( (unsigned int)v39 >= HIDWORD(v39) )
        {
LABEL_62:
          sub_16CD150((__int64)&v38, v40, 0, 8, (int)v11, a6);
          v22 = (unsigned int)v39;
        }
LABEL_31:
        v38[v22] = v19;
        LODWORD(v39) = v39 + 1;
        goto LABEL_18;
      }
LABEL_17:
      sub_16CCBA0((__int64)&v41, v19);
      if ( !v18 )
      {
LABEL_18:
        v16 = *v15;
        goto LABEL_19;
      }
LABEL_30:
      v22 = (unsigned int)v39;
      if ( (unsigned int)v39 >= HIDWORD(v39) )
        goto LABEL_62;
      goto LABEL_31;
    }
    v21 = 0;
    while ( v19 != *v6 )
    {
      if ( *v6 == -2 )
        v21 = v6;
      if ( v20 == ++v6 )
      {
        if ( !v21 )
          goto LABEL_60;
        *v21 = v19;
        --v45;
        ++v41;
        goto LABEL_30;
      }
    }
LABEL_19:
    if ( 16LL * (a2 - 1) != v17 )
    {
      v10 = v43;
      v6 = v42;
      v17 += 16;
      continue;
    }
    break;
  }
  v28 = v15;
  v29 = 0;
  v30 = v28;
  while ( !(unsigned __int8)sub_1D15B50(*(_QWORD *)(v16 + v29), (__int64)&v41, (__int64)&v38, v35, 0, a6) )
  {
    v29 += 16;
    if ( v29 == 16LL * a2 )
      goto LABEL_65;
    v16 = *v30;
  }
  v31 = 0;
LABEL_66:
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  if ( v43 != v42 )
    _libc_free((unsigned __int64)v43);
  return v31;
}
