// Function: sub_2FF8170
// Address: 0x2ff8170
//
__int64 __fastcall sub_2FF8170(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned int a5)
{
  unsigned int v11; // r12d
  __int64 v12; // rax
  _WORD *v13; // rdi
  __int64 v14; // rdx
  _BYTE *v15; // rax
  _BYTE *v16; // rsi
  __int16 *v17; // rax
  int v18; // r8d
  int v19; // r13d
  _WORD *v20; // rax
  __int64 v21; // rdi
  _WORD *v22; // rsi
  __int64 v23; // rdx
  _BYTE *v24; // rax
  _BYTE *v25; // rcx
  unsigned int *v26; // rax
  unsigned int *v27; // rcx
  unsigned int v28; // edx
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r12
  int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // [rsp+8h] [rbp-48h]
  unsigned int v37; // [rsp+Ch] [rbp-44h]
  unsigned int v38; // [rsp+Ch] [rbp-44h]

  if ( *(_BYTE *)(a1 + 72) )
    return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 1128LL))(*(_QWORD *)(a1 + 200));
  v36 = sub_2FF8080(a1, a2, 1);
  v37 = sub_2FE09D0(*(_QWORD *)(a1 + 200), a1, a2);
  if ( sub_2FF7B70(a1) || sub_2FF7B90(a1) )
  {
    if ( sub_2FF7B90(a1) )
    {
      if ( a4 )
      {
        v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 200)
                                                                                           + 1120LL))(
                *(_QWORD *)(a1 + 200),
                a1 + 80,
                a2,
                a3,
                a4,
                a5);
        v11 = v12;
        if ( BYTE4(v12) )
          return v11;
      }
      else
      {
        v30 = *(_QWORD *)(a1 + 184);
        if ( v30 )
        {
          v31 = v30 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL);
          v32 = *(unsigned __int16 *)(v31 + 6) + a3;
          if ( *(unsigned __int16 *)(v31 + 8) > (unsigned int)v32 )
            return *(unsigned int *)(*(_QWORD *)(a1 + 168) + 4 * v32);
        }
      }
      v11 = v37;
      if ( v36 >= v37 )
        return v36;
      return v11;
    }
    v13 = sub_2FF7DB0(a1, a2);
    if ( a3 )
    {
      v14 = a3 - 1;
      v15 = *(_BYTE **)(a2 + 32);
      a3 = 0;
      v16 = &v15[40 * v14];
      while ( 1 )
      {
        if ( !*v15 )
          a3 -= ((v15[3] & 0x10) == 0) - 1;
        if ( v16 == v15 )
          break;
        v15 += 40;
      }
    }
    if ( (unsigned __int16)v13[4] <= a3 )
    {
      v33 = *(unsigned __int16 *)(a2 + 68);
      if ( (_WORD)v33 )
      {
        v34 = (unsigned int)(v33 - 9);
        if ( (unsigned __int16)v34 > 0x3Bu || (v35 = 0x800000000000C09LL, !_bittest64(&v35, v34)) )
        {
          v11 = v37;
          if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x10) == 0 )
            return v11;
        }
      }
    }
    else
    {
      v17 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 184LL) + 4LL * (a3 + (unsigned __int16)v13[3]));
      v18 = *v17;
      if ( v18 < 0 )
        v18 = 1000;
      v38 = v18;
      v11 = v18;
      if ( !a4 )
        return v11;
      v19 = (unsigned __int16)v17[1];
      v20 = sub_2FF7DB0(a1, a4);
      v21 = (unsigned __int16)v20[6];
      v22 = v20;
      if ( !(_WORD)v21 )
        return v11;
      if ( a5 )
      {
        v23 = a5 - 1;
        v24 = *(_BYTE **)(a4 + 32);
        a5 = 0;
        v25 = &v24[40 * v23];
        while ( 1 )
        {
          if ( !*v24 && (v24[4] & 1) == 0 && (v24[4] & 2) == 0 )
            a5 += (v24[3] & 0x10) == 0;
          if ( v25 == v24 )
            break;
          v24 += 40;
        }
      }
      v26 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 192LL) + 12LL * (unsigned __int16)v22[5]);
      v27 = &v26[3 * v21];
      while ( 1 )
      {
        if ( *v26 >= a5 )
        {
          if ( *v26 > a5 )
            return v11;
          v28 = v26[1];
          if ( !v28 || v19 == v28 )
            break;
        }
        v26 += 3;
        if ( v27 == v26 )
          return v11;
      }
      v29 = v26[2];
      if ( v29 <= 0 || v29 <= v38 )
        return v38 - v29;
    }
    return 0;
  }
  else
  {
    return v37;
  }
}
