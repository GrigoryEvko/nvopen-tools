// Function: sub_8CAA20
// Address: 0x8caa20
//
__int64 __fastcall sub_8CAA20(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r14
  _QWORD *v6; // r15
  _QWORD *n; // r12
  _QWORD *v8; // rdx
  __int64 v9; // rdx
  _QWORD *ii; // r12
  _QWORD *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdx
  char v15; // di
  _QWORD *i; // r12
  __int64 v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  _QWORD *j; // r12
  __int64 v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *k; // r12
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *m; // r12
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  result = *a1;
  v3 = *(_QWORD **)(*a1 + 88);
  if ( (__int64 *)v3[13] != a1 )
    return result;
  result = *(unsigned __int8 *)(result + 80);
  v6 = *(_QWORD **)(*a2 + 88LL);
  switch ( (_BYTE)result )
  {
    case 0x13:
      for ( i = (_QWORD *)v3[21]; i; i = (_QWORD *)*i )
      {
        v17 = *(_QWORD *)(i[1] + 88LL);
        v18 = *(_QWORD **)(v17 + 32);
        if ( !v18 || *v18 == v17 && v18[1] != v17 )
        {
          v41 = i[1];
          v19 = sub_878440();
          v20 = qword_4F60240;
          qword_4F60240 = (__int64)v19;
          *v19 = v20;
          v19[1] = v41;
        }
      }
      for ( j = (_QWORD *)v6[21]; j; j = (_QWORD *)*j )
      {
        v23 = *(_QWORD *)(j[1] + 88LL);
        v24 = *(_QWORD **)(v23 + 32);
        if ( !v24 || *v24 == v23 && v24[1] != v23 )
        {
          v42 = j[1];
          v25 = sub_878440();
          v26 = qword_4F60240;
          qword_4F60240 = (__int64)v25;
          *v25 = v26;
          v25[1] = v42;
        }
      }
      result = v3[22];
      if ( result )
      {
        v27 = *(_QWORD *)(result + 88);
        v28 = v6[22];
        if ( !v28 || a2[25] == a1[25] )
          return sub_8CA0A0(v27, 1u);
        else
          return sub_8CA500(v27, *(_QWORD *)(v28 + 88));
      }
      break;
    case 0x14:
      for ( k = (_QWORD *)v3[21]; k; k = (_QWORD *)*k )
      {
        v30 = *(_QWORD *)(k[3] + 88LL);
        v31 = *(_QWORD **)(v30 + 32);
        if ( !v31 || v30 == *v31 && v30 != v31[1] )
        {
          v43 = k[3];
          v32 = sub_878440();
          v33 = qword_4F60240;
          qword_4F60240 = (__int64)v32;
          *v32 = v33;
          v32[1] = v43;
        }
      }
      for ( m = (_QWORD *)v6[21]; m; m = (_QWORD *)*m )
      {
        v35 = *(_QWORD *)(m[3] + 88LL);
        v36 = *(_QWORD **)(v35 + 32);
        if ( !v36 || v35 == *v36 && v35 != v36[1] )
        {
          v44 = m[3];
          v37 = sub_878440();
          v38 = qword_4F60240;
          qword_4F60240 = (__int64)v37;
          *v37 = v38;
          v37[1] = v44;
        }
      }
      v21 = v3[22];
      result = a1[25];
      if ( a2[25] != result )
        return sub_8CC0D0(v21, v6[22]);
      if ( !*(_QWORD *)(v21 + 32) )
      {
        v13 = v3[22];
        v15 = 11;
        return (__int64)sub_8C7090(v15, v13);
      }
      break;
    case 0x15:
      for ( n = (_QWORD *)v3[23]; n; n = (_QWORD *)*n )
      {
        while ( 1 )
        {
          result = *(_QWORD *)(n[1] + 88LL);
          v8 = *(_QWORD **)(result + 32);
          if ( !v8 || *v8 == result && v8[1] != result )
            break;
          n = (_QWORD *)*n;
          if ( !n )
            goto LABEL_13;
        }
        v39 = n[1];
        result = (__int64)sub_878440();
        v9 = qword_4F60240;
        qword_4F60240 = result;
        *(_QWORD *)result = v9;
        *(_QWORD *)(result + 8) = v39;
      }
LABEL_13:
      for ( ii = (_QWORD *)v6[23]; ii; ii = (_QWORD *)*ii )
      {
        while ( 1 )
        {
          result = *(_QWORD *)(ii[1] + 88LL);
          v11 = *(_QWORD **)(result + 32);
          if ( !v11 || *v11 == result && v11[1] != result )
            break;
          ii = (_QWORD *)*ii;
          if ( !ii )
            goto LABEL_20;
        }
        v40 = ii[1];
        result = (__int64)sub_878440();
        v12 = qword_4F60240;
        qword_4F60240 = result;
        *(_QWORD *)result = v12;
        *(_QWORD *)(result + 8) = v40;
      }
LABEL_20:
      v13 = v3[24];
      if ( v13 )
      {
        v14 = v6[24];
        v15 = 7;
        if ( v14 && a2[25] != a1[25] )
          return sub_8CBB20(7, v13, v14);
        return (__int64)sub_8C7090(v15, v13);
      }
      break;
    default:
      sub_721090();
  }
  return result;
}
