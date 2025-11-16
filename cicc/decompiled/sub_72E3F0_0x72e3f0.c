// Function: sub_72E3F0
// Address: 0x72e3f0
//
__int64 __fastcall sub_72E3F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // ebx
  int v8; // eax
  int i; // r14d
  __int64 v10; // rdi
  __int64 result; // rax
  int v12; // r14d
  __int64 v13; // rdi
  _QWORD *v14; // r12
  int v15; // r15d
  int v16; // eax
  char v17; // dl
  int v18; // r14d
  int v19; // eax
  int v20; // edx
  __int64 v21; // rcx
  int v22; // eax
  int v23; // r14d
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdi
  int v30; // r14d
  __int64 v31; // [rsp+8h] [rbp-38h]

  v7 = 0;
  while ( 1 )
  {
    v8 = *(unsigned __int8 *)(a1 + 140);
    for ( i = 0; (_BYTE)v8 == 12; v8 = *(unsigned __int8 *)(a1 + 140) )
    {
      if ( (*(_BYTE *)(a1 + 186) & 8) != 0 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 24LL);
        if ( v10 )
          i += sub_72A8B0(v10, a2, a3, a4, a5, a6);
      }
      a1 = *(_QWORD *)(a1 + 160);
    }
    a3 = (unsigned __int8)v8;
    switch ( (char)v8 )
    {
      case 2:
        v17 = *(_BYTE *)(a1 + 161);
        v18 = i + 13 * *(unsigned __int8 *)(a1 + 160) + 53;
        result = (unsigned int)(v7 + v18);
        if ( (v17 & 8) != 0 )
          return v7 + (unsigned int)sub_72E220(a1, a2) + v18 + 4 * ((v17 & 0x10) != 0) + 2;
        return result;
      case 3:
      case 4:
      case 5:
        return v7 + i + (unsigned int)*(unsigned __int8 *)(a1 + 160) + 87;
      case 6:
        return v7
             + (*(_BYTE *)(a1 + 168) & 1)
             + i
             + (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 160))
             + 107
             + 2 * ((*(_BYTE *)(a1 + 168) & 2) != 0);
      case 7:
        v13 = *(_QWORD *)(a1 + 160);
        v31 = *(_QWORD *)(a1 + 168);
        if ( v13 )
          i += sub_72E3F0(v13);
        v14 = *(_QWORD **)v31;
        if ( *(_QWORD *)v31 )
        {
          v15 = 2;
          do
          {
            v16 = sub_72E3F0(v14[1]);
            v14 = (_QWORD *)*v14;
            i = v15 * (v16 + i);
            ++v15;
          }
          while ( v14 );
        }
        a1 = *(_QWORD *)(v31 + 40);
        result = (unsigned int)(v7 + i);
        if ( !a1 )
          return result;
        v7 += i;
        break;
      case 8:
        v12 = i + sub_72E3F0(*(_QWORD *)(a1 + 160)) + 307;
        result = (unsigned int)(v7 + v12);
        if ( (*(_WORD *)(a1 + 168) & 0x180) == 0 )
          return (unsigned int)(v7 + *(_DWORD *)(a1 + 176) + v12);
        return result;
      case 9:
      case 10:
      case 11:
        return v7 + (unsigned int)sub_72E370((_QWORD *)a1, a2) + i;
      case 13:
        v19 = sub_72E3F0(*(_QWORD *)(a1 + 168));
        a1 = *(_QWORD *)(a1 + 160);
        v7 += v19 + i;
        continue;
      case 14:
        v20 = *(unsigned __int8 *)(a1 + 160);
        v21 = *(_QWORD *)(a1 + 40);
        v22 = i + v20 + (*(_BYTE *)(a1 + 161) & 1) + 499;
        if ( v21 )
          v22 += *(_DWORD *)(v21 + 24);
        v23 = v22 + *(_DWORD *)(a1 + 64) + *(unsigned __int16 *)(a1 + 68);
        result = (unsigned int)(v7 + v23);
        if ( !(_BYTE)v20 )
          return (unsigned int)(v7
                              + *(_DWORD *)(*(_QWORD *)(a1 + 168) + 24LL)
                              + (*(_DWORD *)(*(_QWORD *)(a1 + 168) + 28LL) << 8)
                              + v23);
        return result;
      case 15:
        v24 = sub_72E3F0(*(_QWORD *)(a1 + 160));
        v29 = *(_QWORD *)(a1 + 168);
        v30 = i + v24 + 331;
        result = (unsigned int)(v7 + v30);
        if ( v29 )
          return v7 + 5 * (unsigned int)sub_72DB90(v29, a2, v25, v26, v27, v28) + v30;
        return result;
      default:
        return (unsigned int)(v7 + v8 + i);
    }
  }
}
