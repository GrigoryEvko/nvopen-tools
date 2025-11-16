// Function: sub_C699F0
// Address: 0xc699f0
//
_QWORD *__fastcall sub_C699F0(unsigned int a1, _QWORD *a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  unsigned int v6; // r13d
  char v7; // r14
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rdx
  __int64 v11; // r12
  _QWORD *v12; // rdx
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rcx
  _QWORD *v15; // rdx
  __int64 v16; // r12
  _QWORD *v17; // rdx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rcx
  _QWORD *v20; // rdx
  __int64 v21; // r12
  _QWORD *v22; // rdx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rax
  _QWORD *result; // rax
  __int64 v26; // r13
  _QWORD *v27; // rax
  unsigned __int64 v28; // r9
  unsigned __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // r12
  _QWORD *v32; // rax
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rdx
  __int64 v35; // r13
  _QWORD *v36; // rax
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdx
  __int64 v39; // r15
  _QWORD *v40; // rdx
  unsigned __int64 v41; // r9
  unsigned __int64 v42; // rcx
  _QWORD *v43; // rdx
  _QWORD *v44; // [rsp+10h] [rbp-40h]
  _QWORD *v45; // [rsp+18h] [rbp-38h]

  if ( a1 <= 0x7F )
  {
    v35 = a2[1];
    v36 = (_QWORD *)*a2;
    v37 = v35 + 1;
    if ( (_QWORD *)*a2 == a2 + 2 )
      v38 = 15;
    else
      v38 = a2[2];
    if ( v37 > v38 )
    {
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v36 = (_QWORD *)*a2;
    }
    *((_BYTE *)v36 + v35) = a1;
    result = (_QWORD *)*a2;
    a2[1] = v37;
    *((_BYTE *)result + v35 + 1) = 0;
  }
  else if ( a1 <= 0x7FF )
  {
    v26 = a2[1];
    v27 = (_QWORD *)*a2;
    v28 = v26 + 1;
    if ( (_QWORD *)*a2 == a2 + 2 )
      v29 = 15;
    else
      v29 = a2[2];
    if ( v28 > v29 )
    {
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v27 = (_QWORD *)*a2;
      v28 = v26 + 1;
    }
    *((_BYTE *)v27 + v26) = (a1 >> 6) | 0xC0;
    v30 = (_QWORD *)*a2;
    a2[1] = v28;
    *((_BYTE *)v30 + v26 + 1) = 0;
    v31 = a2[1];
    v32 = (_QWORD *)*a2;
    v33 = v31 + 1;
    if ( (_QWORD *)*a2 == a2 + 2 )
      v34 = 15;
    else
      v34 = a2[2];
    if ( v33 > v34 )
    {
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v32 = (_QWORD *)*a2;
    }
    *((_BYTE *)v32 + v31) = a1 & 0x3F | 0x80;
    result = (_QWORD *)*a2;
    a2[1] = v33;
    *((_BYTE *)result + v31 + 1) = 0;
  }
  else
  {
    if ( a1 <= 0xFFFF )
    {
      v39 = a2[1];
      v40 = (_QWORD *)*a2;
      v5 = a2 + 2;
      v7 = a1 & 0x3F | 0x80;
      v41 = v39 + 1;
      v6 = (a1 >> 6) & 0x3F | 0xFFFFFF80;
      if ( (_QWORD *)*a2 == a2 + 2 )
        v42 = 15;
      else
        v42 = a2[2];
      if ( v41 > v42 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v40 = (_QWORD *)*a2;
        v5 = a2 + 2;
        v41 = v39 + 1;
      }
      *((_BYTE *)v40 + v39) = (a1 >> 12) | 0xE0;
      v43 = (_QWORD *)*a2;
      a2[1] = v41;
      *((_BYTE *)v43 + v39 + 1) = 0;
    }
    else
    {
      if ( a1 > 0x10FFFF )
        BUG();
      v3 = a2[1];
      v4 = (_QWORD *)*a2;
      v5 = a2 + 2;
      v6 = (a1 >> 6) & 0x3F | 0xFFFFFF80;
      v7 = a1 & 0x3F | 0x80;
      v8 = v3 + 1;
      if ( (_QWORD *)*a2 == a2 + 2 )
        v9 = 15;
      else
        v9 = a2[2];
      if ( v8 > v9 )
      {
        sub_2240BB0(a2, v3, 0, 0, 1);
        v4 = (_QWORD *)*a2;
        v5 = a2 + 2;
        v8 = v3 + 1;
      }
      *((_BYTE *)v4 + v3) = (a1 >> 18) | 0xF0;
      v10 = (_QWORD *)*a2;
      a2[1] = v8;
      *((_BYTE *)v10 + v3 + 1) = 0;
      v11 = a2[1];
      v12 = (_QWORD *)*a2;
      v13 = v11 + 1;
      if ( (_QWORD *)*a2 == v5 )
        v14 = 15;
      else
        v14 = a2[2];
      if ( v13 > v14 )
      {
        v44 = v5;
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v12 = (_QWORD *)*a2;
        v5 = v44;
        v13 = v11 + 1;
      }
      *((_BYTE *)v12 + v11) = (a1 >> 12) & 0x3F | 0x80;
      v15 = (_QWORD *)*a2;
      a2[1] = v13;
      *((_BYTE *)v15 + v11 + 1) = 0;
    }
    v16 = a2[1];
    v17 = (_QWORD *)*a2;
    v18 = v16 + 1;
    if ( (_QWORD *)*a2 == v5 )
      v19 = 15;
    else
      v19 = a2[2];
    if ( v18 > v19 )
    {
      v45 = v5;
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v17 = (_QWORD *)*a2;
      v5 = v45;
    }
    *((_BYTE *)v17 + v16) = v6;
    v20 = (_QWORD *)*a2;
    a2[1] = v18;
    *((_BYTE *)v20 + v16 + 1) = 0;
    v21 = a2[1];
    v22 = (_QWORD *)*a2;
    v23 = v21 + 1;
    if ( (_QWORD *)*a2 == v5 )
      v24 = 15;
    else
      v24 = a2[2];
    if ( v23 > v24 )
    {
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v22 = (_QWORD *)*a2;
    }
    *((_BYTE *)v22 + v21) = v7;
    result = (_QWORD *)*a2;
    a2[1] = v23;
    *((_BYTE *)result + v21 + 1) = 0;
  }
  return result;
}
