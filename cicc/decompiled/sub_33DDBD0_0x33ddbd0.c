// Function: sub_33DDBD0
// Address: 0x33ddbd0
//
__int64 __fastcall sub_33DDBD0(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        unsigned int a7)
{
  unsigned int v7; // r14d
  unsigned int v9; // r15d
  unsigned int *v10; // rdx
  __int64 v11; // rax
  unsigned __int16 v12; // r12
  __int64 v13; // rax
  unsigned int *v15; // r12
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  unsigned __int16 *v21; // rax
  unsigned __int16 v22; // bx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // ebx
  _QWORD *v26; // rax
  __int64 v27; // rax
  int *v28; // rax
  int *v29; // rsi
  unsigned int v30; // ecx
  unsigned __int64 v31; // rdx
  unsigned int v32; // ebx
  unsigned __int64 v33; // r13
  unsigned __int64 *v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // [rsp+0h] [rbp-90h]
  __int64 v38; // [rsp+8h] [rbp-88h]
  unsigned __int16 v39; // [rsp+10h] [rbp-80h] BYREF
  __int64 v40; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v41; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-68h]
  unsigned __int64 *v43; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-58h]
  unsigned __int64 v45; // [rsp+40h] [rbp-50h] BYREF
  __int64 v46; // [rsp+48h] [rbp-48h]
  unsigned __int64 v47; // [rsp+50h] [rbp-40h]
  unsigned int v48; // [rsp+58h] [rbp-38h]

  v7 = a3;
  if ( !a6 || (*(_DWORD *)(a2 + 28) & 0x407F) == 0 )
  {
    v9 = *(_DWORD *)(a2 + 24);
    switch ( v9 )
    {
      case 0x34u:
      case 0x36u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x52u:
      case 0x53u:
      case 0x54u:
      case 0x55u:
      case 0x9Cu:
      case 0x9Fu:
      case 0xA0u:
      case 0xA8u:
      case 0xACu:
      case 0xADu:
      case 0xB4u:
      case 0xB5u:
      case 0xB6u:
      case 0xB7u:
      case 0xBAu:
      case 0xBBu:
      case 0xBCu:
      case 0xC1u:
      case 0xC2u:
      case 0xC3u:
      case 0xC4u:
      case 0xC5u:
      case 0xC8u:
      case 0xC9u:
      case 0xCAu:
      case 0xD5u:
      case 0xD6u:
      case 0xD8u:
      case 0xDEu:
      case 0xE0u:
      case 0xE1u:
      case 0xEAu:
        return 0;
      case 0x9Du:
      case 0x9Eu:
        v10 = *(unsigned int **)(a2 + 40);
        v11 = *(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16LL * v10[2];
        v12 = *(_WORD *)v11;
        v40 = *(_QWORD *)(v11 + 8);
        v13 = 20;
        if ( v9 != 157 )
          v13 = 10;
        v39 = v12;
        v37 = *(_QWORD *)&v10[v13 + 2];
        v38 = *(_QWORD *)&v10[v13];
        v9 = sub_33DE230(a1, v38, v37, a4, a5, a7 + 1);
        if ( !(_BYTE)v9 )
          return 1;
        sub_33DD090((__int64)&v45, (__int64)a1, v38, v37, a7 + 1);
        v32 = v46;
        v44 = v46;
        if ( (unsigned int)v46 > 0x40 )
        {
          sub_C43780((__int64)&v43, (const void **)&v45);
          v32 = v44;
          if ( v44 > 0x40 )
          {
            sub_C43D10((__int64)&v43);
            v32 = v44;
            v34 = v43;
            goto LABEL_48;
          }
          v33 = (unsigned __int64)v43;
        }
        else
        {
          v33 = v45;
        }
        v34 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & ~v33);
        if ( !v32 )
          v34 = 0;
LABEL_48:
        v42 = v32;
        v41 = v34;
        if ( v12 )
        {
          LODWORD(v35) = word_4456340[v12 - 1];
        }
        else
        {
          v35 = sub_3007240((__int64)&v39);
          v43 = (unsigned __int64 *)v35;
        }
        v36 = (unsigned int)v35;
        if ( v32 <= 0x40 )
        {
          LOBYTE(v9) = (unsigned int)v35 <= (unsigned __int64)v34;
          goto LABEL_52;
        }
        if ( v32 - (unsigned int)sub_C444A0((__int64)&v41) > 0x40 || v36 <= *v34 )
        {
          if ( !v34 )
            goto LABEL_52;
        }
        else
        {
          v9 = 0;
        }
        j_j___libc_free_0_0((unsigned __int64)v34);
LABEL_52:
        if ( v48 > 0x40 && v47 )
          j_j___libc_free_0_0(v47);
        if ( (unsigned int)v46 > 0x40 && v45 )
          j_j___libc_free_0_0(v45);
        return v9;
      case 0xA5u:
        v21 = *(unsigned __int16 **)(a2 + 48);
        v22 = *v21;
        v23 = *((_QWORD *)v21 + 1);
        LOWORD(v45) = v22;
        v46 = v23;
        if ( v22 )
        {
          if ( (unsigned __int16)(v22 - 176) <= 0x34u )
          {
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            sub_CA17B0(
              "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se MVT::getVectorElementCount() instead");
          }
          v24 = word_4456340[v22 - 1];
        }
        else
        {
          if ( sub_3007100((__int64)&v45) )
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
          v24 = (unsigned int)sub_3007130((__int64)&v45, a2);
        }
        v28 = *(int **)(a2 + 96);
        v29 = &v28[v24];
        if ( v29 == v28 )
          return 0;
        v30 = 0;
        break;
      case 0xA7u:
        if ( a5 )
          return 0;
        v25 = *(_DWORD *)(a4 + 8);
        if ( v25 <= 0x40 )
        {
          v26 = *(_QWORD **)a4;
        }
        else
        {
          if ( v25 - (unsigned int)sub_C444A0(a4) > 0x40 )
            return 1;
          v26 = **(_QWORD ***)a4;
        }
        LOBYTE(v9) = (unsigned __int64)v26 > 1;
        return v9;
      case 0xBEu:
      case 0xBFu:
      case 0xC0u:
        v9 = 1;
        if ( (unsigned __int8)sub_33DE230(
                                a1,
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                a4,
                                a5,
                                a7 + 1) )
        {
          v19 = sub_33DCFD0((__int64)a1, a2, v7, a4, a7 + 1);
          v46 = v20;
          v45 = (unsigned __int64)v19;
          return (unsigned __int8)v20 ^ 1u;
        }
        return v9;
      case 0xCFu:
      case 0xD0u:
        v15 = *(unsigned int **)(a2 + 40);
        v16 = *(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * v15[2];
        v17 = *(_WORD *)v16;
        v18 = *(_QWORD *)(v16 + 8);
        LOWORD(v45) = v17;
        v46 = v18;
        if ( v17 )
        {
          if ( (unsigned __int16)(v17 - 2) <= 7u
            || (unsigned __int16)(v17 - 17) <= 0x6Cu
            || (unsigned __int16)(v17 - 176) <= 0x1Fu )
          {
            return 0;
          }
        }
        else if ( sub_3007070((__int64)&v45) )
        {
          return 0;
        }
        v27 = 20;
        if ( v9 != 208 )
          v27 = 40;
        if ( (*(_BYTE *)(*(_QWORD *)&v15[v27] + 96LL) & 0x10) == 0 )
        {
          LOBYTE(v9) = ((*a1)[108] & 6) != 0;
          return v9;
        }
        return 1;
      default:
        if ( v9 - 46 > 2 && v9 <= 0x1F3 )
          return 1;
        return (*(unsigned int (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD **, _QWORD, _QWORD, _QWORD))(*a1[2] + 2088LL))(
                 a1[2],
                 a2,
                 a3,
                 a4,
                 a1,
                 a5,
                 a6,
                 a7);
    }
    while ( 1 )
    {
      if ( *v28 < 0 )
      {
        v31 = *(_QWORD *)a4;
        if ( *(_DWORD *)(a4 + 8) > 0x40u )
          v31 = *(_QWORD *)(v31 + 8LL * (v30 >> 6));
        if ( (v31 & (1LL << v30)) != 0 )
          break;
      }
      ++v28;
      ++v30;
      if ( v29 == v28 )
        return 0;
    }
  }
  return 1;
}
