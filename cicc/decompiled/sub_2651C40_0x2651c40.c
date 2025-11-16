// Function: sub_2651C40
// Address: 0x2651c40
//
__int64 __fastcall sub_2651C40(
        _QWORD *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // r11
  __int64 v14; // r10
  signed __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 result; // rax
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 i; // r13
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // r14
  _QWORD *v25; // rbx
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+8h] [rbp-88h]
  signed __int64 v33; // [rsp+10h] [rbp-80h]
  signed __int64 v34; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  __int64 v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  signed __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+20h] [rbp-70h]
  signed __int64 v41; // [rsp+20h] [rbp-70h]
  __int64 v42; // [rsp+20h] [rbp-70h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+20h] [rbp-70h]
  int v45; // [rsp+20h] [rbp-70h]
  __int64 v46; // [rsp+28h] [rbp-68h]
  __int64 v47; // [rsp+30h] [rbp-60h]
  _QWORD *v48; // [rsp+38h] [rbp-58h]
  __int64 v49; // [rsp+48h] [rbp-48h]
  __int64 v50[7]; // [rsp+58h] [rbp-38h] BYREF

  while ( 1 )
  {
    v8 = (_QWORD *)a6;
    v49 = a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( a4 <= v9 )
      break;
    v10 = a5;
    if ( a5 <= a7 )
    {
      result = sub_26410C0(a2, a3, a6);
      v50[0] = a8;
      if ( a1 == a2 )
        return sub_2640F50((__int64)v8, result, v49);
      if ( v8 != (_QWORD *)result )
      {
        v19 = (__int64)(a2 - 9);
        v20 = result - 72;
        for ( i = v49 - 72; ; i -= 72 )
        {
          if ( sub_2650F70(v50, v20, v19) )
          {
            sub_2641BF0(i, v19);
            if ( a1 == (_QWORD *)v19 )
              return sub_2640F50((__int64)v8, v20 + 72, i);
            v19 -= 72;
          }
          else
          {
            result = sub_2641BF0(i, v20);
            if ( v8 == (_QWORD *)v20 )
              return result;
            v20 -= 72;
          }
        }
      }
      return result;
    }
    v11 = a4;
    if ( a4 <= a5 )
    {
      v43 = a5 / 2;
      v46 = (__int64)&a2[9 * (a5 / 2)];
      v30 = sub_2651630((__int64)a1, (__int64)a2, v46, a8);
      v14 = a7;
      v15 = v43;
      v13 = (_QWORD *)v30;
      v47 = 0x8E38E38E38E38E39LL * ((v30 - (__int64)a1) >> 3);
    }
    else
    {
      v47 = a4 / 2;
      v40 = (__int64)&a1[9 * (a4 / 2)];
      v12 = sub_2651750((__int64)a2, a3, v40, a8);
      v13 = (_QWORD *)v40;
      v14 = a7;
      v46 = v12;
      v15 = 0x8E38E38E38E38E39LL * ((v12 - (__int64)a2) >> 3);
    }
    v16 = v11 - v47;
    if ( v16 <= v15 || v15 > v14 )
    {
      if ( v16 > v14 )
      {
        v35 = v14;
        v39 = v15;
        v45 = (int)v13;
        v17 = sub_2646940((__int64)v13, (__int64)a2, v46);
        v14 = v35;
        v15 = v39;
        LODWORD(v13) = v45;
      }
      else
      {
        v17 = v46;
        if ( v16 )
        {
          v31 = v14;
          v33 = v15;
          v42 = (__int64)v13;
          v37 = sub_26410C0(v13, (__int64)a2, (__int64)v8);
          sub_26410C0(a2, v46, v42);
          v17 = sub_2640F50((__int64)v8, v37, v46);
          LODWORD(v13) = v42;
          v15 = v33;
          v14 = v31;
        }
      }
    }
    else
    {
      v17 = (__int64)v13;
      if ( v15 )
      {
        v34 = v15;
        v32 = v14;
        v44 = (__int64)v13;
        v38 = sub_26410C0(a2, v46, (__int64)v8);
        sub_2640F50(v44, (__int64)a2, v46);
        v17 = sub_26410C0(v8, v38, v44);
        LODWORD(v13) = v44;
        v15 = v34;
        v14 = v32;
      }
    }
    v36 = v14;
    v41 = v15;
    v48 = (_QWORD *)v17;
    sub_2651C40((_DWORD)a1, (_DWORD)v13, v17, v47, v15, (_DWORD)v8, v14, a8);
    a6 = (__int64)v8;
    a4 = v16;
    a2 = (_QWORD *)v46;
    a7 = v36;
    a3 = v49;
    a5 = v10 - v41;
    a1 = v48;
  }
  v22 = (__int64)a2;
  v23 = (__int64)a1;
  v24 = sub_26410C0(a1, (__int64)a2, a6);
  result = a8;
  v50[0] = a8;
  if ( (_QWORD *)v24 != v8 )
  {
    v25 = v8;
    do
    {
      while ( 1 )
      {
        if ( v49 == v22 )
          return sub_26410C0(v25, v24, v23);
        if ( !sub_2650F70(v50, v22, (__int64)v25) )
          break;
        v26 = v22;
        v27 = v23;
        v22 += 72;
        v23 += 72;
        result = sub_2641BF0(v27, v26);
        if ( (_QWORD *)v24 == v25 )
          return result;
      }
      v28 = (__int64)v25;
      v29 = v23;
      v25 += 9;
      v23 += 72;
      result = sub_2641BF0(v29, v28);
    }
    while ( (_QWORD *)v24 != v25 );
  }
  return result;
}
