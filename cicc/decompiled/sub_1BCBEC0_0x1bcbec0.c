// Function: sub_1BCBEC0
// Address: 0x1bcbec0
//
__int64 __fastcall sub_1BCBEC0(__int64 a1, __int64 ***a2, unsigned int a3)
{
  __int64 *v6; // rdi
  unsigned int v7; // ebx
  _QWORD *v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rcx
  __int64 ***v11; // r12
  int v12; // r13d
  __int64 **v13; // rdx
  int v14; // r11d
  __int64 ***v15; // r10
  unsigned int v16; // eax
  __int64 ***v17; // r8
  __int64 **v18; // rdi
  unsigned int v19; // ecx
  unsigned int *v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // ecx
  __int64 **v23; // rdi
  int v24; // eax
  int v25; // r11d
  __int64 ***v26; // r9
  unsigned int v27; // r12d
  __int64 ***v29; // r9
  int v30; // r11d
  unsigned int v31; // ecx
  int v32; // r10d
  unsigned int *v33; // r9
  int v34; // edx
  __int64 v35; // rsi
  unsigned int v36; // ecx
  int v37; // r10d
  unsigned int *v38; // r8
  int v39; // r10d
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 v42; // [rsp+10h] [rbp-70h] BYREF
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+20h] [rbp-60h]
  __int64 v45; // [rsp+28h] [rbp-58h]
  __int64 v46; // [rsp+30h] [rbp-50h] BYREF
  __int64 v47; // [rsp+38h] [rbp-48h]
  __int64 v48; // [rsp+40h] [rbp-40h]
  __int64 v49; // [rsp+48h] [rbp-38h]

  v6 = **a2;
  if ( *((_BYTE *)*a2 + 16) == 55 )
    v6 = (__int64 *)**(*a2 - 6);
  v7 = a3;
  v42 = 0;
  v43 = 0;
  v8 = sub_16463B0(v6, a3);
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  if ( a3 )
  {
    v9 = 0;
    v10 = 0;
    v11 = &a2[a3 - 1];
    v12 = 37 * a3 - 37;
    while ( 1 )
    {
      --v7;
      if ( !v9 )
        break;
      v13 = *v11;
      v14 = 1;
      v15 = 0;
      v16 = (v9 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
      v17 = (__int64 ***)(v10 + 8LL * v16);
      v18 = *v17;
      if ( *v11 == *v17 )
      {
LABEL_6:
        if ( (_DWORD)v45 )
        {
          v19 = v12 & (v45 - 1);
          v20 = (unsigned int *)(v43 + 4LL * v19);
          v21 = *v20;
          if ( v7 == *v20 )
            goto LABEL_8;
          v32 = 1;
          v33 = 0;
          while ( v21 != -1 )
          {
            if ( v33 || v21 != -2 )
              v20 = v33;
            v19 = (v45 - 1) & (v32 + v19);
            v21 = *(_DWORD *)(v43 + 4LL * v19);
            if ( v7 == v21 )
              goto LABEL_8;
            ++v32;
            v33 = v20;
            v20 = (unsigned int *)(v43 + 4LL * v19);
          }
          if ( !v33 )
            v33 = v20;
          ++v42;
          v34 = v44 + 1;
          if ( 4 * ((int)v44 + 1) < (unsigned int)(3 * v45) )
          {
            if ( (int)v45 - HIDWORD(v44) - v34 <= (unsigned int)v45 >> 3 )
            {
              sub_136B240((__int64)&v42, v45);
              if ( !(_DWORD)v45 )
              {
LABEL_84:
                LODWORD(v44) = v44 + 1;
                BUG();
              }
              v39 = 1;
              v38 = 0;
              LODWORD(v40) = v12 & (v45 - 1);
              v33 = (unsigned int *)(v43 + 4LL * (unsigned int)v40);
              v41 = *v33;
              v34 = v44 + 1;
              if ( v7 != *v33 )
              {
                while ( v41 != -1 )
                {
                  if ( v41 == -2 && !v38 )
                    v38 = v33;
                  v40 = ((_DWORD)v45 - 1) & (unsigned int)(v40 + v39);
                  v33 = (unsigned int *)(v43 + 4 * v40);
                  v41 = *v33;
                  if ( v7 == *v33 )
                    goto LABEL_44;
                  ++v39;
                }
                goto LABEL_60;
              }
            }
            goto LABEL_44;
          }
        }
        else
        {
          ++v42;
        }
        sub_136B240((__int64)&v42, 2 * v45);
        if ( !(_DWORD)v45 )
          goto LABEL_84;
        LODWORD(v35) = v12 & (v45 - 1);
        v33 = (unsigned int *)(v43 + 4LL * (unsigned int)v35);
        v36 = *v33;
        v34 = v44 + 1;
        if ( v7 != *v33 )
        {
          v37 = 1;
          v38 = 0;
          while ( v36 != -1 )
          {
            if ( !v38 && v36 == -2 )
              v38 = v33;
            v35 = ((_DWORD)v45 - 1) & (unsigned int)(v35 + v37);
            v33 = (unsigned int *)(v43 + 4 * v35);
            v36 = *v33;
            if ( v7 == *v33 )
              goto LABEL_44;
            ++v37;
          }
LABEL_60:
          if ( v38 )
            v33 = v38;
        }
LABEL_44:
        LODWORD(v44) = v34;
        if ( *v33 != -1 )
          --HIDWORD(v44);
        *v33 = v7;
LABEL_8:
        --v11;
        v12 -= 37;
        if ( !v7 )
          goto LABEL_31;
        goto LABEL_9;
      }
      while ( v18 != (__int64 **)-8LL )
      {
        if ( v15 || v18 != (__int64 **)-16LL )
          v17 = v15;
        v16 = (v9 - 1) & (v14 + v16);
        v18 = *(__int64 ***)(v10 + 8LL * v16);
        if ( v13 == v18 )
          goto LABEL_6;
        ++v14;
        v15 = v17;
        v17 = (__int64 ***)(v10 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
      ++v46;
      v24 = v48 + 1;
      if ( 4 * ((int)v48 + 1) >= 3 * v9 )
        goto LABEL_12;
      if ( v9 - (v24 + HIDWORD(v48)) <= v9 >> 3 )
      {
        sub_1353F00((__int64)&v46, v9);
        if ( !(_DWORD)v49 )
        {
LABEL_85:
          LODWORD(v48) = v48 + 1;
          BUG();
        }
        v29 = 0;
        v30 = 1;
        v31 = (v49 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
        v15 = (__int64 ***)(v47 + 8LL * v31);
        v13 = *v15;
        v24 = v48 + 1;
        if ( *v11 != *v15 )
        {
          while ( v13 != (__int64 **)-8LL )
          {
            if ( !v29 && v13 == (__int64 **)-16LL )
              v29 = v15;
            v31 = (v49 - 1) & (v30 + v31);
            v15 = (__int64 ***)(v47 + 8LL * v31);
            v13 = *v15;
            if ( *v11 == *v15 )
              goto LABEL_28;
            ++v30;
          }
          v13 = *v11;
          if ( v29 )
            v15 = v29;
        }
      }
LABEL_28:
      LODWORD(v48) = v24;
      if ( *v15 != (__int64 **)-8LL )
        --HIDWORD(v48);
      *v15 = v13;
      --v11;
      v12 -= 37;
      if ( !v7 )
        goto LABEL_31;
LABEL_9:
      v10 = v47;
      v9 = v49;
    }
    ++v46;
LABEL_12:
    sub_1353F00((__int64)&v46, 2 * v9);
    if ( !(_DWORD)v49 )
      goto LABEL_85;
    v13 = *v11;
    v22 = (v49 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
    v15 = (__int64 ***)(v47 + 8LL * v22);
    v23 = *v15;
    v24 = v48 + 1;
    if ( *v15 != *v11 )
    {
      v25 = 1;
      v26 = 0;
      while ( v23 != (__int64 **)-8LL )
      {
        if ( v23 == (__int64 **)-16LL && !v26 )
          v26 = v15;
        v22 = (v49 - 1) & (v25 + v22);
        v15 = (__int64 ***)(v47 + 8LL * v22);
        v23 = *v15;
        if ( v13 == *v15 )
          goto LABEL_28;
        ++v25;
      }
      if ( v26 )
        v15 = v26;
    }
    goto LABEL_28;
  }
LABEL_31:
  v27 = sub_1BBDB90(a1, (__int64)v8, (__int64)&v42);
  j___libc_free_0(v47);
  j___libc_free_0(v43);
  return v27;
}
