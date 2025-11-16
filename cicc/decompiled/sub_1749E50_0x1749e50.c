// Function: sub_1749E50
// Address: 0x1749e50
//
__int64 __fastcall sub_1749E50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        double a8)
{
  int v8; // eax
  __int64 v9; // rbx
  __int64 *v10; // r15
  unsigned int v11; // r12d
  __int64 v12; // r14
  __int64 **v13; // r13
  unsigned int v14; // ecx
  _QWORD *v15; // rax
  int v16; // ebx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 *v24; // rax
  _QWORD *v25; // rax
  __int64 **v26; // rax
  unsigned int v27; // eax
  unsigned __int8 v28; // bl
  _QWORD *v29; // rax
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 *v32; // r9
  __int64 *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r12d
  __int64 **v37; // r9
  __int64 *v38; // r15
  unsigned int v39; // ebx
  unsigned int v40; // [rsp+4h] [rbp-4Ch]
  unsigned int v41; // [rsp+8h] [rbp-48h]
  int v42; // [rsp+Ch] [rbp-44h]
  __int64 **v43; // [rsp+10h] [rbp-40h]
  unsigned int v44; // [rsp+10h] [rbp-40h]
  unsigned __int8 v45; // [rsp+18h] [rbp-38h]
  unsigned int v46; // [rsp+1Ch] [rbp-34h]

  while ( 1 )
  {
    v8 = *(unsigned __int8 *)(a1 + 16);
    v45 = a5;
    if ( (_BYTE)v8 == 9 )
      return 1;
    v9 = *(_QWORD *)a1;
    v10 = (__int64 *)a1;
    v11 = a2;
    v12 = a3;
    v13 = (__int64 **)a4;
    if ( a4 == *(_QWORD *)a1 )
      break;
    if ( (unsigned __int8)v8 > 0x10u )
    {
      v21 = *(_QWORD *)(a1 + 8);
      if ( !v21 || *(_QWORD *)(v21 + 8) || (unsigned __int8)v8 <= 0x17u )
        return 0;
      v22 = v8 - 24;
      if ( v22 == 37 )
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v37 = *(__int64 ***)(a1 - 8);
        else
          v37 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        v38 = *v37;
        v39 = sub_1643030(**v37);
        if ( v39 % (unsigned int)sub_1643030((__int64)v13) )
          return 0;
        a5 = v45;
        a4 = (__int64)v13;
        a3 = v12;
        a2 = (unsigned int)a2;
        a1 = (__int64)v38;
      }
      else if ( v22 <= 0x25 )
      {
        if ( v22 == 23 )
        {
          if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
            v33 = *(__int64 **)(a1 - 8);
          else
            v33 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
          v34 = v33[3];
          if ( *(_BYTE *)(v34 + 16) != 13 )
            return 0;
          v35 = *(_DWORD *)(v34 + 32) <= 0x40u ? *(_QWORD *)(v34 + 24) : **(_QWORD **)(v34 + 24);
          v36 = v35 + a2;
          if ( ((int)v35 + (int)a2) % (unsigned int)sub_1643030(a4) )
            return 0;
          a5 = v45;
          a1 = *v33;
          a4 = (__int64)v13;
          a3 = v12;
          a2 = v36;
        }
        else
        {
          if ( v22 != 27 )
            return 0;
          v28 = a5;
          v29 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
              ? *(_QWORD **)(a1 - 8)
              : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
          if ( !(unsigned __int8)sub_1749E50(*v29, (unsigned int)a2, v12, a4, a5) )
            return 0;
          if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
            v30 = *(_QWORD *)(a1 - 8);
          else
            v30 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          a1 = *(_QWORD *)(v30 + 24);
          a5 = v28;
          a4 = (__int64)v13;
          a3 = v12;
          a2 = (unsigned int)a2;
        }
      }
      else
      {
        if ( v22 != 47 )
          return 0;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v32 = *(__int64 **)(a1 - 8);
        else
          v32 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        a1 = *v32;
        a3 = v12;
        a2 = (unsigned int)a2;
      }
    }
    else
    {
      v41 = sub_1643030(*(_QWORD *)a1);
      v14 = sub_1643030((__int64)v13);
      v42 = v41 / v14;
      if ( v41 / v14 != 1 )
      {
        v46 = v14;
        if ( *(_BYTE *)(v9 + 8) != 11 )
        {
          v44 = v14;
          v25 = (_QWORD *)sub_16498A0(a1);
          v26 = (__int64 **)sub_1644900(v25, v41);
          v10 = (__int64 *)sub_15A4510((__int64 ***)a1, v26, 0);
          v27 = sub_1643030((__int64)v13);
          v14 = v44;
          v46 = v27;
        }
        v40 = v14;
        v15 = (_QWORD *)sub_16498A0((__int64)v10);
        v43 = (__int64 **)sub_1644900(v15, v46);
        if ( v41 >= v40 )
        {
          v16 = 0;
          while ( 1 )
          {
            v17 = sub_15A0680(*v10, v11, 0);
            v18 = sub_15A2D80(v10, v17, 0, a6, a7, a8);
            v19 = sub_15A43B0(v18, v43, 0);
            if ( !(unsigned __int8)sub_1749E50(v19, v11, v12, v13, v45) )
              break;
            ++v16;
            v11 += v46;
            if ( v42 == v16 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
      v31 = sub_15A4510((__int64 ***)a1, v13, 0);
      a5 = v45;
      a4 = (__int64)v13;
      a3 = v12;
      a1 = v31;
      a2 = (unsigned int)a2;
    }
  }
  if ( (unsigned __int8)v8 <= 0x10u && sub_1593BB0(a1, a2, a3, a4) )
    return 1;
  v23 = (unsigned int)a2 / (unsigned int)sub_1643030(v9);
  if ( v45 )
    v23 = *(_DWORD *)(v12 + 8) + ~v23;
  v24 = (__int64 *)(*(_QWORD *)v12 + 8LL * v23);
  if ( !*v24 )
  {
    *v24 = a1;
    return 1;
  }
  return 0;
}
