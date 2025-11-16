// Function: sub_F09E10
// Address: 0xf09e10
//
__int64 __fastcall sub_F09E10(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 v9; // r15
  unsigned int v10; // r13d
  __int64 v11; // r12
  __int64 *v12; // r10
  __int64 v13; // r15
  unsigned int **v14; // r12
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // r15d
  bool v20; // al
  __int64 v21; // r15
  __int64 v22; // rcx
  __int64 *v23; // r12
  __int64 v24; // r10
  unsigned int **v25; // r12
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rdx
  bool v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rax
  unsigned __int8 *v32; // rcx
  unsigned int v33; // r15d
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // r14
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // r13
  _BYTE *v40; // rax
  unsigned int v41; // r13d
  bool v42; // al
  __int64 v43; // r15
  __int64 v44; // rdx
  _BYTE *v45; // rax
  unsigned int v46; // r15d
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 i; // r14
  __int64 v50; // rdx
  unsigned int v51; // esi
  bool v52; // r13
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned int v55; // edx
  unsigned int v56; // r13d
  int v57; // eax
  int v58; // [rsp+0h] [rbp-A0h]
  int v59; // [rsp+4h] [rbp-9Ch]
  __int64 *v60; // [rsp+8h] [rbp-98h]
  __int64 v61; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v62; // [rsp+8h] [rbp-98h]
  __int64 v63; // [rsp+8h] [rbp-98h]
  __int64 v64; // [rsp+8h] [rbp-98h]
  __int64 v65; // [rsp+8h] [rbp-98h]
  unsigned int v66; // [rsp+8h] [rbp-98h]
  _BYTE v67[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v68; // [rsp+30h] [rbp-70h]
  const char *v69; // [rsp+40h] [rbp-60h] BYREF
  __int64 v70; // [rsp+48h] [rbp-58h]
  __int16 v71; // [rsp+60h] [rbp-40h]

  result = 0;
  if ( **(_DWORD **)a1 != 13 )
    return result;
  v6 = *(_QWORD **)(a1 + 8);
  result = **(_QWORD **)(a1 + 16);
  if ( *v6 )
  {
    if ( result || *a3 != 44 )
      return 0;
    v18 = *((_QWORD *)a3 - 8);
    if ( *(_BYTE *)v18 == 17 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
        v20 = *(_QWORD *)(v18 + 24) == 0;
      else
        v20 = v19 == (unsigned int)sub_C444A0(v18 + 24);
    }
    else
    {
      v43 = *(_QWORD *)(v18 + 8);
      v44 = (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17;
      if ( (unsigned int)v44 > 1 || *(_BYTE *)v18 > 0x15u )
        return 0;
      v63 = *((_QWORD *)a3 - 8);
      v45 = sub_AD7630(v63, 0, v44);
      v32 = (unsigned __int8 *)v63;
      if ( !v45 || *v45 != 17 )
      {
        if ( *(_BYTE *)(v43 + 8) == 17 )
        {
          v58 = *(_DWORD *)(v43 + 32);
          if ( v58 )
          {
            v29 = 0;
            v30 = 0;
            while ( 1 )
            {
              v62 = v32;
              v31 = sub_AD69F0(v32, v30);
              v32 = v62;
              if ( !v31 )
                break;
              if ( *(_BYTE *)v31 != 13 )
              {
                if ( *(_BYTE *)v31 != 17 )
                  break;
                v33 = *(_DWORD *)(v31 + 32);
                if ( v33 <= 0x40 )
                {
                  v29 = *(_QWORD *)(v31 + 24) == 0;
                }
                else
                {
                  v34 = sub_C444A0(v31 + 24);
                  v32 = v62;
                  v29 = v33 == v34;
                }
                if ( !v29 )
                  break;
              }
              v30 = (unsigned int)(v30 + 1);
              if ( v58 == (_DWORD)v30 )
              {
                if ( v29 )
                  goto LABEL_19;
                goto LABEL_36;
              }
            }
          }
        }
        goto LABEL_36;
      }
      v46 = *((_DWORD *)v45 + 8);
      if ( v46 <= 0x40 )
        v20 = *((_QWORD *)v45 + 3) == 0;
      else
        v20 = v46 == (unsigned int)sub_C444A0((__int64)(v45 + 24));
    }
    if ( v20 )
    {
LABEL_19:
      v21 = *((_QWORD *)a3 - 4);
      if ( v21 )
      {
        v22 = *((_QWORD *)a3 - 4);
        v23 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 32LL);
        v68 = 257;
        v24 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v23[10] + 32LL))(
                v23[10],
                15,
                a4,
                v22,
                0,
                0);
        if ( !v24 )
        {
          v71 = 257;
          v64 = sub_B504D0(15, a4, v21, (__int64)&v69, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v23[11] + 16LL))(
            v23[11],
            v64,
            v67,
            v23[7],
            v23[8]);
          v24 = v64;
          v47 = 16LL * *((unsigned int *)v23 + 2);
          v48 = *v23;
          for ( i = v48 + v47; i != v48; v24 = v65 )
          {
            v50 = *(_QWORD *)(v48 + 8);
            v51 = *(_DWORD *)v48;
            v48 += 16;
            v65 = v24;
            sub_B99FD0(v24, v51, v50);
          }
        }
        v61 = v24;
        v25 = *(unsigned int ***)(*(_QWORD *)(a1 + 24) + 32LL);
        v69 = sub_BD5D20(*(_QWORD *)(a1 + 40));
        v26 = *(__int64 **)(a1 + 32);
        v70 = v27;
        v28 = *(__int64 **)(a1 + 8);
        v71 = 261;
        return sub_B36550(v25, *v26, *v28, v61, (__int64)&v69, 0);
      }
    }
LABEL_36:
    if ( !**(_QWORD **)(a1 + 16) || *a2 != 44 )
      return 0;
    goto LABEL_9;
  }
  if ( !result )
    return result;
  if ( *a2 != 44 )
    return 0;
LABEL_9:
  v9 = *((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v9 == 17 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 > 0x40 )
    {
      if ( v10 == (unsigned int)sub_C444A0(v9 + 24) )
        goto LABEL_12;
      return 0;
    }
    if ( *(_QWORD *)(v9 + 24) )
      return 0;
  }
  else
  {
    v39 = *(_QWORD *)(v9 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 > 1 || *(_BYTE *)v9 > 0x15u )
      return 0;
    v40 = sub_AD7630(*((_QWORD *)a2 - 8), 0, (__int64)v6);
    if ( !v40 || *v40 != 17 )
    {
      if ( *(_BYTE *)(v39 + 8) == 17 )
      {
        v59 = *(_DWORD *)(v39 + 32);
        if ( v59 )
        {
          v52 = 0;
          v53 = 0;
          while ( 1 )
          {
            v66 = v53;
            v54 = sub_AD69F0((unsigned __int8 *)v9, v53);
            if ( !v54 )
              break;
            v55 = v66;
            if ( *(_BYTE *)v54 != 13 )
            {
              if ( *(_BYTE *)v54 != 17 )
                break;
              v56 = *(_DWORD *)(v54 + 32);
              if ( v56 <= 0x40 )
              {
                v52 = *(_QWORD *)(v54 + 24) == 0;
              }
              else
              {
                v57 = sub_C444A0(v54 + 24);
                v55 = v66;
                v52 = v56 == v57;
              }
              if ( !v52 )
                break;
            }
            v53 = v55 + 1;
            if ( v59 == v53 )
            {
              if ( v52 )
                goto LABEL_12;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v41 = *((_DWORD *)v40 + 8);
    if ( v41 <= 0x40 )
      v42 = *((_QWORD *)v40 + 3) == 0;
    else
      v42 = v41 == (unsigned int)sub_C444A0((__int64)(v40 + 24));
    if ( !v42 )
      return 0;
  }
LABEL_12:
  v11 = *((_QWORD *)a2 - 4);
  if ( !v11 )
    return 0;
  v12 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 32LL);
  v68 = 257;
  v60 = v12;
  v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v12[10] + 32LL))(
          v12[10],
          15,
          a4,
          v11,
          0,
          0);
  if ( !v13 )
  {
    v71 = 257;
    v13 = sub_B504D0(15, a4, v11, (__int64)&v69, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v60[11] + 16LL))(
      v60[11],
      v13,
      v67,
      v60[7],
      v60[8]);
    v35 = *v60;
    v36 = *v60 + 16LL * *((unsigned int *)v60 + 2);
    if ( *v60 != v36 )
    {
      do
      {
        v37 = *(_QWORD *)(v35 + 8);
        v38 = *(_DWORD *)v35;
        v35 += 16;
        sub_B99FD0(v13, v38, v37);
      }
      while ( v36 != v35 );
    }
  }
  v14 = *(unsigned int ***)(*(_QWORD *)(a1 + 24) + 32LL);
  v69 = sub_BD5D20(*(_QWORD *)(a1 + 40));
  v15 = *(__int64 **)(a1 + 32);
  v70 = v16;
  v17 = *(__int64 **)(a1 + 16);
  v71 = 261;
  return sub_B36550(v14, *v15, v13, *v17, (__int64)&v69, 0);
}
