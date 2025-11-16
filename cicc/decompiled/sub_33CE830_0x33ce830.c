// Function: sub_33CE830
// Address: 0x33ce830
//
__int64 __fastcall sub_33CE830(_QWORD **a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned int v7; // eax
  _QWORD *v8; // r14
  _QWORD *v9; // r12
  unsigned int v10; // ebx
  unsigned int v11; // r15d
  __int64 v13; // rbx
  _BYTE *v14; // rdi
  __int64 *v15; // rax
  __int64 *v16; // rdx
  unsigned int v17; // ebx
  unsigned int v18; // r14d
  __int64 v19; // rax
  unsigned int v20; // ebx
  __int64 *v21; // rax
  unsigned int v22; // ebx
  unsigned int v23; // r14d
  __int64 v24; // rax
  char v25; // [rsp-30h] [rbp-30h]
  __int64 v26; // [rsp-28h] [rbp-28h]
  __int64 v27; // [rsp-18h] [rbp-18h]

  while ( 1 )
  {
    if ( ((*a1)[108] & 4) != 0 )
      return 1;
    v27 = v6;
    v26 = v5;
    if ( (*(_BYTE *)(a2 + 28) & 0x20) != 0 )
      return 1;
    if ( a5 > 5 )
      return 0;
    v7 = *(_DWORD *)(a2 + 24);
    if ( v7 == 36 || v7 == 12 )
      break;
    switch ( v7 )
    {
      case 0x60u:
      case 0x61u:
      case 0x62u:
      case 0x63u:
      case 0x64u:
      case 0x96u:
      case 0x97u:
      case 0xF6u:
      case 0xF8u:
      case 0xF9u:
      case 0xFAu:
      case 0xFBu:
      case 0xFCu:
      case 0xFDu:
      case 0xFEu:
      case 0xFFu:
      case 0x100u:
      case 0x101u:
      case 0x102u:
      case 0x104u:
      case 0x106u:
      case 0x107u:
      case 0x108u:
        return a4;
      case 0x98u:
      case 0x9Eu:
      case 0xF4u:
      case 0xF5u:
        v16 = *(__int64 **)(a2 + 40);
        a4 = (unsigned __int8)a4;
        ++a5;
        a2 = *v16;
        a3 = v16[1];
        goto LABEL_26;
      case 0x9Au:
      case 0xE6u:
      case 0xE9u:
      case 0x103u:
      case 0x109u:
      case 0x10Au:
      case 0x10Bu:
      case 0x10Cu:
      case 0x10Du:
      case 0x10Eu:
      case 0x10Fu:
      case 0x110u:
      case 0x111u:
      case 0x112u:
      case 0x113u:
      case 0x114u:
      case 0x115u:
      case 0x116u:
        if ( (_BYTE)a4 )
          return 1;
        v15 = *(__int64 **)(a2 + 40);
        ++a5;
        a4 = 0;
        a2 = *v15;
        a3 = v15[1];
        goto LABEL_26;
      case 0x9Cu:
        v8 = *(_QWORD **)(a2 + 40);
        v9 = &v8[5 * *(unsigned int *)(a2 + 64)];
        if ( v9 == v8 )
          return 1;
        v10 = a5 + 1;
        v11 = (unsigned __int8)a4;
        while ( (unsigned __int8)sub_33CE830(a1, *v8, v8[1], v11, v10) )
        {
          v8 += 5;
          if ( v9 == v8 )
            return 1;
        }
        return 0;
      case 0xCDu:
        v22 = a5 + 1;
        v23 = (unsigned __int8)a4;
        if ( !(unsigned __int8)sub_33CE830(
                                 a1,
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                 (unsigned __int8)a4,
                                 a5 + 1) )
          return 0;
        v24 = *(_QWORD *)(a2 + 40);
        a5 = v22;
        a4 = v23;
        a2 = *(_QWORD *)(v24 + 80);
        a3 = *(_QWORD *)(v24 + 88);
        goto LABEL_26;
      case 0xDCu:
      case 0xDDu:
        return 1;
      case 0x117u:
      case 0x118u:
      case 0x11Du:
      case 0x11Eu:
        v17 = a5 + 1;
        v18 = (unsigned __int8)a4;
        if ( (unsigned __int8)sub_33CE830(
                                a1,
                                **(_QWORD **)(a2 + 40),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                (unsigned __int8)a4,
                                a5 + 1) )
          return 1;
        goto LABEL_29;
      case 0x119u:
      case 0x11Au:
        if ( (_BYTE)a4 )
          return 1;
        v20 = a5 + 1;
        if ( (unsigned __int8)sub_33CE830(
                                a1,
                                **(_QWORD **)(a2 + 40),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                0,
                                a5 + 1) )
        {
          if ( (unsigned __int8)sub_33CE830(
                                  a1,
                                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                  1,
                                  v20) )
            return 1;
        }
        if ( !(unsigned __int8)sub_33CE830(
                                 a1,
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                 0,
                                 v20) )
          return 0;
        v21 = *(__int64 **)(a2 + 40);
        a5 = v20;
        a4 = 1;
        a2 = *v21;
        a3 = v21[1];
        v5 = v26;
        continue;
      case 0x11Bu:
      case 0x11Cu:
        v17 = a5 + 1;
        v18 = (unsigned __int8)a4;
        if ( !(unsigned __int8)sub_33CE830(
                                 a1,
                                 **(_QWORD **)(a2 + 40),
                                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                 (unsigned __int8)a4,
                                 a5 + 1) )
          return 0;
LABEL_29:
        v19 = *(_QWORD *)(a2 + 40);
        a5 = v17;
        a4 = v18;
        a2 = *(_QWORD *)(v19 + 40);
        a3 = *(_QWORD *)(v19 + 48);
LABEL_26:
        v5 = v26;
        v6 = v27;
        break;
      default:
        if ( v7 - 46 > 2 && v7 <= 0x1F3 )
          return 0;
        return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, _QWORD, _QWORD))(*a1[2] + 2104LL))(
                 a1[2],
                 a2,
                 a3,
                 a1,
                 (unsigned __int8)a4,
                 a5);
    }
  }
  v13 = *(_QWORD *)(a2 + 96);
  v25 = a4;
  v14 = (_BYTE *)(v13 + 24);
  if ( *(void **)(v13 + 24) == sub_C33340() )
    v14 = *(_BYTE **)(v13 + 32);
  if ( (v14[20] & 7) != 1 )
    return 1;
  if ( v25 )
    return (unsigned int)sub_C35FD0(v14) ^ 1;
  return 0;
}
