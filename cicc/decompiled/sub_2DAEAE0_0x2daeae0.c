// Function: sub_2DAEAE0
// Address: 0x2daeae0
//
__int64 __fastcall sub_2DAEAE0(__int64 *a1, int a2)
{
  __int64 v3; // r13
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rax
  _DWORD *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r12
  int v20; // esi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // rdx
  char v33; // al
  int v34; // r8d
  _QWORD *v35; // r12
  _DWORD *v36; // rax
  _QWORD *v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned int v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  unsigned int v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+28h] [rbp-38h]
  unsigned int v47; // [rsp+28h] [rbp-38h]
  __int64 v48; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  if ( !sub_2DADE10(*a1, a2) )
    return -1;
  if ( a2 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(v3 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(v3 + 304) + 8LL * (unsigned int)a2);
  if ( v7 )
  {
    if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
    {
      v7 = *(_QWORD *)(v7 + 32);
      if ( v7 )
      {
        if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
LABEL_60:
          BUG();
      }
    }
  }
  v8 = *(_QWORD *)(v7 + 16);
  v9 = *(unsigned __int16 *)(v8 + 68);
  if ( (unsigned __int16)v9 > 0x14u || ((1LL << v9) & 0x180301) == 0 )
  {
    if ( (_WORD)v9 == 10 )
      return 0;
    v11 = *(unsigned __int8 *)(v7 + 3);
    v10 = (unsigned __int8)v11 >> 4;
    LOBYTE(v11) = (unsigned __int8)v11 >> 6;
    if ( (v10 & 1 & (unsigned __int8)v11) != 0 )
      return 0;
    else
      return sub_2EBF1E0(v3, (unsigned int)a2, v11, v9, v4, v5);
  }
  v12 = a2 & 0x7FFFFFFF;
  v13 = 8LL * ((a2 & 0x7FFFFFFFu) >> 6);
  *(_QWORD *)(v13 + a1[22]) |= 1LL << a2;
  v14 = (_QWORD *)(a1[13] + v13);
  if ( (*v14 & (1LL << a2)) == 0 )
  {
    *v14 |= 1LL << a2;
    v15 = (_DWORD *)a1[9];
    if ( v15 == (_DWORD *)(a1[11] - 4) )
    {
      v35 = (_QWORD *)a1[12];
      if ( (((__int64)v15 - a1[10]) >> 2) + (((((__int64)v35 - a1[8]) >> 3) - 1) << 7) + ((a1[7] - a1[5]) >> 2) == 0x1FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( (unsigned __int64)(a1[4] - (((__int64)v35 - a1[3]) >> 3)) <= 1 )
      {
        sub_1D7F850(a1 + 3, 1u, 0);
        v35 = (_QWORD *)a1[12];
      }
      v35[1] = sub_22077B0(0x200u);
      v36 = (_DWORD *)a1[9];
      if ( v36 )
        *v36 = v12;
      v37 = (_QWORD *)(a1[12] + 8);
      a1[12] = (__int64)v37;
      v38 = (_QWORD *)*v37;
      v39 = *v37 + 512LL;
      a1[10] = (__int64)v38;
      a1[11] = v39;
      a1[9] = (__int64)v38;
    }
    else
    {
      if ( v15 )
      {
        *v15 = v12;
        v15 = (_DWORD *)a1[9];
      }
      a1[9] = (__int64)(v15 + 1);
    }
  }
  result = 0;
  if ( (((*(_BYTE *)(v7 + 3) & 0x40) != 0) & (*(_BYTE *)(v7 + 3) >> 4)) == 0 )
  {
    v16 = *(_QWORD *)(v8 + 32);
    v40 = *(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16 * v12) & 0xFFFFFFFFFFFFFFF8LL;
    v17 = v16 + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
    v18 = 5LL * (unsigned int)sub_2E88FE0(v8);
    if ( v17 != v16 + 8 * v18 )
    {
      v41 = 0;
      v19 = v16 + 8 * v18;
      v42 = 0;
      while ( 1 )
      {
        if ( *(_BYTE *)v19 )
          goto LABEL_34;
        v33 = *(_BYTE *)(v19 + 4);
        if ( (v33 & 1) != 0 || (v33 & 2) != 0 || (*(_BYTE *)(v19 + 3) & 0x10) != 0 && (*(_DWORD *)v19 & 0xFFF00) == 0 )
          goto LABEL_34;
        v34 = *(_DWORD *)(v19 + 8);
        if ( !v34 )
          goto LABEL_34;
        if ( (unsigned int)(v34 - 1) <= 0x3FFFFFFE )
          break;
        v45 = *(_DWORD *)(v19 + 8);
        if ( sub_2DADC20((_QWORD *)*a1, v8, v40, (_DWORD *)v19) )
          break;
        v20 = v45;
        v43 = v45;
        v46 = *a1;
        v24 = *a1;
        if ( !sub_2DADE10(*a1, v20) )
          goto LABEL_31;
        if ( (v43 & 0x80000000) != 0 )
        {
          v25 = *(_QWORD *)(*(_QWORD *)(v46 + 56) + 16LL * (v43 & 0x7FFFFFFF) + 8);
        }
        else
        {
          v21 = *(_QWORD *)(v46 + 304);
          v25 = *(_QWORD *)(v21 + 8LL * v43);
        }
        if ( v25 )
        {
          if ( (*(_BYTE *)(v25 + 3) & 0x10) == 0 )
          {
            v25 = *(_QWORD *)(v25 + 32);
            if ( v25 )
            {
              if ( (*(_BYTE *)(v25 + 3) & 0x10) == 0 )
                goto LABEL_60;
            }
          }
        }
        v22 = *(unsigned __int16 *)(*(_QWORD *)(v25 + 16) + 68LL);
        if ( ((unsigned __int16)v22 > 0x14u || ((1LL << v22) & 0x180301) == 0) && (_WORD)v22 != 10 )
        {
LABEL_31:
          v47 = (*(_DWORD *)v19 >> 8) & 0xFFF;
          v26 = sub_2EBF1E0(v24, v43, v21, v22, v43, v23);
          v28 = v26;
          v29 = v27;
          if ( v47 )
          {
            v28 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)a1[1] + 320LL))(
                    a1[1],
                    v47,
                    v26,
                    v27);
            v29 = v30;
          }
          goto LABEL_33;
        }
LABEL_34:
        v19 += 40;
        if ( v17 == v19 )
          return v42;
      }
      v29 = -1;
      v28 = -1;
LABEL_33:
      v44 = v29;
      v48 = v28;
      v31 = sub_2EAB0A0(v19);
      v42 |= sub_2DAE4E0(a1, v7, v31, v48, v44);
      v41 |= v32;
      goto LABEL_34;
    }
    return 0;
  }
  return result;
}
