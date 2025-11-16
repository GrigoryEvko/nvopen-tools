// Function: sub_291A860
// Address: 0x291a860
//
_QWORD *__fastcall sub_291A860(__int64 a1, __int64 *a2, unsigned __int64 a3, unsigned __int64 a4)
{
  char v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdx
  char v13; // r15
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  unsigned __int64 v16; // r15
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  _QWORD *v20; // r12
  char v22; // r15
  __int64 v23; // rdx
  __int64 v24; // r15
  char v25; // al
  char v26; // al
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // al
  _QWORD *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  unsigned int v39; // [rsp+14h] [rbp-5Ch]
  __int64 v40; // [rsp+18h] [rbp-58h]
  unsigned int v41; // [rsp+18h] [rbp-58h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  unsigned __int64 v43; // [rsp+20h] [rbp-50h]
  char v44; // [rsp+28h] [rbp-48h]
  __int64 v45; // [rsp+30h] [rbp-40h] BYREF
  __int64 v46; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    while ( 1 )
    {
      if ( a3 )
      {
        v22 = sub_AE5020(a1, (__int64)a2);
        v45 = sub_9208B0(a1, (__int64)a2);
        v46 = v23;
        if ( a3 > (((unsigned __int64)(v45 + 7) >> 3) + (1LL << v22) - 1) >> v22 << v22 )
          return 0;
      }
      else
      {
        v8 = sub_AE5020(a1, (__int64)a2);
        v9 = (__int64)a2;
        v10 = a1;
        v45 = sub_9208B0(a1, (__int64)a2);
        v46 = v11;
        if ( a4 == ((1LL << v8) + ((unsigned __int64)(v45 + 7) >> 3) - 1) >> v8 << v8 )
          return (_QWORD *)sub_291A050(v10, v9);
        sub_AE5020(a1, (__int64)a2);
        v45 = sub_9208B0(a1, (__int64)a2);
        v46 = v12;
      }
      v13 = sub_AE5020(a1, (__int64)a2);
      v45 = sub_9208B0(a1, (__int64)a2);
      v46 = v14;
      if ( ((((unsigned __int64)(v45 + 7) >> 3) + (1LL << v13) - 1) >> v13 << v13) - a3 < a4 )
        return 0;
      v15 = *((_BYTE *)a2 + 8);
      if ( v15 == 16 )
        break;
      if ( (unsigned int)v15 - 17 <= 1 )
      {
        v16 = *((unsigned int *)a2 + 8);
        a2 = (__int64 *)a2[3];
        goto LABEL_8;
      }
      if ( v15 != 15 )
        return 0;
      v24 = sub_AE4AC0(a1, (__int64)a2);
      if ( *(_BYTE *)(v24 + 8) )
        return 0;
      v45 = *(_QWORD *)v24;
      LOBYTE(v46) = 0;
      if ( sub_CA1930(&v45) <= a3 )
        return 0;
      v25 = *(_BYTE *)(v24 + 8);
      v43 = a3 + a4;
      v45 = *(_QWORD *)v24;
      LOBYTE(v46) = v25;
      if ( sub_CA1930(&v45) < a3 + a4 )
        return 0;
      v39 = sub_AE1C80(v24, a3);
      v26 = *(_BYTE *)(v24 + 16LL * v39 + 32);
      v45 = *(_QWORD *)(v24 + 16LL * v39 + 24);
      LOBYTE(v46) = v26;
      a3 -= sub_CA1930(&v45);
      v37 = 8LL * v39;
      v40 = *(_QWORD *)(a2[2] + v37);
      v27 = sub_BDB740(a1, v40);
      v45 = v27;
      v46 = v28;
      if ( a3 >= v27 )
        return 0;
      v9 = v40;
      if ( !a3 && a4 >= v27 )
      {
        if ( a4 != v27 )
        {
          v41 = *((_DWORD *)a2 + 3);
          v29 = *(_BYTE *)(v24 + 8);
          v30 = (_QWORD *)(a2[2] + v37);
          v38 = a2[2];
          v45 = *(_QWORD *)v24;
          LOBYTE(v46) = v29;
          if ( sub_CA1930(&v45) <= v43 )
          {
            v31 = v38 + 8LL * v41;
            goto LABEL_34;
          }
          v34 = sub_AE1C80(v24, v43);
          if ( v39 == v34 )
            return 0;
          v42 = v34;
          v35 = v24 + 16LL * v34 + 24;
          v36 = *(_QWORD *)v35;
          LOBYTE(v35) = *(_BYTE *)(v35 + 8);
          v45 = v36;
          LOBYTE(v46) = v35;
          if ( sub_CA1930(&v45) != v43 )
            return 0;
          v31 = a2[2] + 8 * v42;
LABEL_34:
          v20 = sub_BD0B90((_QWORD *)*a2, v30, (v31 - (__int64)v30) >> 3, (a2[1] & 0x200) != 0);
          v32 = sub_AE4AC0(a1, (__int64)v20);
          v33 = *(_QWORD *)v32;
          LOBYTE(v32) = *(_BYTE *)(v32 + 8);
          v45 = v33;
          LOBYTE(v46) = v32;
          if ( sub_CA1930(&v45) != a4 )
            return 0;
          return v20;
        }
        goto LABEL_27;
      }
      if ( a4 + a3 > v27 )
        return 0;
      a2 = (__int64 *)v40;
    }
    v16 = a2[4];
    a2 = (__int64 *)a2[3];
LABEL_8:
    v44 = sub_AE5020(a1, (__int64)a2);
    v45 = sub_9208B0(a1, (__int64)a2);
    v46 = v17;
    v18 = (((unsigned __int64)(v45 + 7) >> 3) + (1LL << v44) - 1) >> v44 << v44;
    v19 = a3 % v18;
    if ( v16 <= a3 / v18 )
      return 0;
    a3 %= v18;
    if ( !v19 && a4 >= v18 )
      break;
    if ( a4 + v19 > v18 )
      return 0;
  }
  v9 = (__int64)a2;
  if ( a4 == v18 )
  {
LABEL_27:
    v10 = a1;
    return (_QWORD *)sub_291A050(v10, v9);
  }
  if ( a4 / v18 * v18 != a4 )
    return 0;
  return sub_BCD420(a2, a4 / v18);
}
