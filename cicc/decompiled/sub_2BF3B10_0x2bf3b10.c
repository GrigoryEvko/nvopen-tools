// Function: sub_2BF3B10
// Address: 0x2bf3b10
//
__int64 __fastcall sub_2BF3B10(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r13
  __int64 v5; // rdi
  __int64 result; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  bool v22; // of
  __int64 v23; // rcx
  __int64 v24; // rsi
  char v25; // al
  _QWORD *v26; // r13
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // r13
  __int64 v31; // rax
  _QWORD *v32; // rdi
  int v33; // esi
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  _QWORD *v39; // r14
  _QWORD *v40; // r15
  unsigned __int64 v41; // rsi
  _QWORD *v42; // rax
  _QWORD *v43; // r14
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  int v47; // esi
  _QWORD *v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  _BYTE *v51; // [rsp+0h] [rbp-240h]
  _QWORD *v52; // [rsp+8h] [rbp-238h]
  _QWORD *v53; // [rsp+10h] [rbp-230h]
  __int64 v54; // [rsp+18h] [rbp-228h]
  _QWORD v55[12]; // [rsp+20h] [rbp-220h] BYREF
  __int64 v56; // [rsp+80h] [rbp-1C0h]
  __int64 v57; // [rsp+88h] [rbp-1B8h]
  _QWORD v58[12]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v59; // [rsp+100h] [rbp-140h]
  __int64 v60; // [rsp+108h] [rbp-138h]
  _QWORD v61[15]; // [rsp+120h] [rbp-120h] BYREF
  _BYTE v62[168]; // [rsp+198h] [rbp-A8h] BYREF

  v4 = HIDWORD(a2);
  v54 = a2;
  if ( *(_BYTE *)(a1 + 128) )
  {
    if ( BYTE4(a2) )
      return 0;
    v5 = **(_QWORD **)(*(_QWORD *)(a1 + 112) + 80LL);
    result = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v5 + 24LL))(v5, a2);
    if ( (_DWORD)a2 == 1 && *(_DWORD *)(a3 + 176) != 2 )
      result /= 2;
    return result;
  }
  v52 = v58;
  v58[0] = *(_QWORD *)(a1 + 112);
  v53 = v61;
  sub_2BF3840(v61, v58);
  v7 = 0;
  sub_2ABD910(v55, (__int64)v61, v8, v9, v10, v11);
  v51 = v62;
  sub_2ABD910(v58, (__int64)v62, v12, v13, v14, v15);
  while ( 1 )
  {
    v23 = v56;
    v24 = v59;
    if ( v57 - v56 == v60 - v59 )
      break;
LABEL_7:
    BYTE4(v54) = v4;
    v18 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, _BYTE *, _QWORD *, _QWORD *))(**(_QWORD **)(v57 - 24) + 24LL))(
            *(_QWORD *)(v57 - 24),
            v54,
            a3,
            v23,
            v16,
            v17,
            v51,
            v52,
            v53);
    v22 = __OFADD__(v18, v7);
    v7 += v18;
    if ( v22 )
    {
      v7 = 0x8000000000000000LL;
      if ( v18 > 0 )
        v7 = 0x7FFFFFFFFFFFFFFFLL;
    }
    sub_2ADA290((__int64)v55, v54, v19, 1, v20, v21);
  }
  while ( v57 != v23 )
  {
    if ( *(_QWORD *)v23 != *(_QWORD *)v24 )
      goto LABEL_7;
    v25 = *(_BYTE *)(v23 + 16);
    if ( v25 != *(_BYTE *)(v24 + 16) || v25 && *(_QWORD *)(v23 + 8) != *(_QWORD *)(v24 + 8) )
      goto LABEL_7;
    v23 += 24;
    v24 += 24;
  }
  sub_2AB1B10((__int64)v52);
  sub_2AB1B10((__int64)v55);
  sub_2AB1B10((__int64)v51);
  sub_2AB1B10((__int64)v53);
  v26 = sub_C52410();
  v27 = v26 + 1;
  v28 = sub_C959E0();
  v29 = (_QWORD *)v26[2];
  v30 = v26 + 1;
  if ( v29 )
  {
    do
    {
      while ( 1 )
      {
        v37 = v29[2];
        v38 = v29[3];
        if ( v28 <= v29[4] )
          break;
        v29 = (_QWORD *)v29[3];
        if ( !v38 )
          goto LABEL_36;
      }
      v30 = v29;
      v29 = (_QWORD *)v29[2];
    }
    while ( v37 );
LABEL_36:
    if ( v27 != v30 && v28 < v30[4] )
      v30 = v27;
  }
  if ( v30 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_28;
  v31 = v30[7];
  if ( !v31 )
    goto LABEL_28;
  v32 = v30 + 6;
  v33 = qword_500DDC0[1];
  do
  {
    while ( 1 )
    {
      v34 = *(_QWORD *)(v31 + 16);
      v35 = *(_QWORD *)(v31 + 24);
      if ( *(_DWORD *)(v31 + 32) >= v33 )
        break;
      v31 = *(_QWORD *)(v31 + 24);
      if ( !v35 )
        goto LABEL_26;
    }
    v32 = (_QWORD *)v31;
    v31 = *(_QWORD *)(v31 + 16);
  }
  while ( v34 );
LABEL_26:
  if ( v30 + 6 == v32 || v33 < *((_DWORD *)v32 + 8) || !*((_DWORD *)v32 + 9) )
  {
LABEL_28:
    v36 = sub_DFD270(*(_QWORD *)a3, 2, *(_DWORD *)(a3 + 176));
  }
  else
  {
    v39 = sub_C52410();
    v40 = v39 + 1;
    v41 = sub_C959E0();
    v42 = (_QWORD *)v39[2];
    if ( v42 )
    {
      v43 = v39 + 1;
      do
      {
        while ( 1 )
        {
          v44 = v42[2];
          v45 = v42[3];
          if ( v41 <= v42[4] )
            break;
          v42 = (_QWORD *)v42[3];
          if ( !v45 )
            goto LABEL_46;
        }
        v43 = v42;
        v42 = (_QWORD *)v42[2];
      }
      while ( v44 );
LABEL_46:
      if ( v40 != v43 && v41 < v43[4] )
        v43 = v40;
      if ( v43 == (_QWORD *)((char *)sub_C52410() + 8) )
        return v7;
    }
    else
    {
      if ( v40 == (_QWORD *)((char *)sub_C52410() + 8) )
        return v7;
      v43 = v39 + 1;
    }
    v46 = v43[7];
    if ( !v46 )
      return v7;
    v47 = qword_500DDC0[1];
    v48 = v43 + 6;
    do
    {
      while ( 1 )
      {
        v49 = *(_QWORD *)(v46 + 16);
        v50 = *(_QWORD *)(v46 + 24);
        if ( *(_DWORD *)(v46 + 32) >= v47 )
          break;
        v46 = *(_QWORD *)(v46 + 24);
        if ( !v50 )
          goto LABEL_55;
      }
      v48 = (_QWORD *)v46;
      v46 = *(_QWORD *)(v46 + 16);
    }
    while ( v49 );
LABEL_55:
    if ( v48 == v43 + 6 || v47 < *((_DWORD *)v48 + 8) )
      return v7;
    v36 = *((int *)v48 + 9);
  }
  result = v36 + v7;
  if ( __OFADD__(v36, v7) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v36 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
