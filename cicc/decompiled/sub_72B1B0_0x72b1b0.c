// Function: sub_72B1B0
// Address: 0x72b1b0
//
__int64 __fastcall sub_72B1B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rbp
  __int64 v9; // r12
  __int64 v10; // kr00_8
  _QWORD *v11; // rbx
  __int64 result; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  _QWORD *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r13
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 i; // r12
  _QWORD v28[7]; // [rsp-38h] [rbp-38h] BYREF

  v10 = v6;
  v28[6] = v8;
  v28[3] = v9;
  v28[2] = v7;
  v11 = a2;
  result = *(unsigned __int8 *)(a1 + 24);
  switch ( *(_BYTE *)(a1 + 24) )
  {
    case 0:
    case 8:
    case 0x11:
    case 0x13:
      goto LABEL_4;
    case 1:
      result = *(unsigned __int8 *)(a1 + 56);
      if ( (unsigned __int8)(result - 105) > 4u )
      {
        if ( (_BYTE)result != 19 )
          return result;
        goto LABEL_4;
      }
      v15 = *(_QWORD **)(a1 + 72);
      v16 = sub_72B0F0((__int64)v15, 0);
      v21 = v16;
      if ( v16 )
      {
        result = sub_8D7760(v16, 0, v17, v18, v19, v20);
        if ( (_DWORD)result )
          return result;
        if ( (*(_BYTE *)(v21 + 193) & 1) != 0 )
        {
          result = (__int64)&dword_4F077BC;
          if ( dword_4F077BC )
          {
            result = (unsigned int)qword_4F077B4;
            if ( !(_DWORD)qword_4F077B4 )
            {
              result = (__int64)&qword_4F077A8;
              if ( qword_4F077A8 <= 0x15F8Fu )
              {
                v28[0] = 0;
                v28[1] = 0;
                if ( (unsigned int)sub_7A2E10(a1, v28) )
                  return sub_67E3D0(v28);
                result = sub_67E3D0(v28);
              }
            }
          }
        }
        goto LABEL_4;
      }
      v25 = *v15;
      v26 = *v15;
      if ( (unsigned int)sub_8D2E30(*v15) )
      {
        for ( i = sub_8D46C0(v26); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
      }
      else
      {
        result = sub_8D3D10(v26);
        if ( !(_DWORD)result )
          goto LABEL_4;
        for ( i = sub_8D4870(v25); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
      }
      result = sub_8D2310(i);
      if ( !(_DWORD)result )
        goto LABEL_4;
      result = sub_8D76D0(i);
      if ( !(_DWORD)result )
        goto LABEL_4;
      return result;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 9:
    case 0xA:
    case 0x10:
    case 0x12:
      return result;
    case 7:
      v13 = *(_QWORD *)(a1 + 56);
      v14 = *(_QWORD *)(v13 + 16);
      if ( v14 )
        goto LABEL_8;
      v22 = sub_691620(*(_QWORD *)(v13 + 8));
      if ( (*(_BYTE *)v13 & 1) != 0 )
      {
        result = sub_8D3410(*(_QWORD *)(v13 + 8));
        if ( !(_DWORD)result )
          return result;
        result = sub_691630(v22, 1);
        if ( !(_DWORD)result )
          return result;
        v24 = sub_7D3810(unk_4D04844 == 0 ? 1 : 3);
        a2 = v28;
        result = sub_87AC70(v24, v28);
        v14 = *(_QWORD *)(result + 88);
      }
      else if ( (*(_BYTE *)v13 & 8) != 0 && (a2 = 0, (unsigned int)sub_691630(v22, 0)) )
      {
        v23 = sub_7D3810(unk_4D04844 == 0 ? 2 : 4);
        a2 = (_QWORD *)v22;
        result = sub_87C270(v23, v22, v28);
        v14 = *(_QWORD *)(result + 88);
      }
      else
      {
        result = sub_8D3A70(v22);
        if ( !(_DWORD)result )
          return result;
        while ( *(_BYTE *)(v22 + 140) == 12 )
          v22 = *(_QWORD *)(v22 + 160);
        result = *(_QWORD *)(v22 + 168);
        v14 = *(_QWORD *)(result + 184);
      }
      if ( v14 )
      {
LABEL_8:
        result = sub_8D7760(v14, a2, a3, a4, a5, a6);
        if ( !(_DWORD)result )
        {
LABEL_4:
          *((_DWORD *)v11 + 20) = 1;
          *((_DWORD *)v11 + 18) = 1;
        }
      }
      return result;
    case 0xB:
      result = *(_QWORD *)(a1 + 56);
      if ( *(_QWORD *)(result + 16) )
      {
        if ( (*(_BYTE *)(a1 + 64) & 1) != 0 )
          goto LABEL_4;
        result = sub_8DD3B0(*(_QWORD *)(result + 56));
        if ( (_DWORD)result )
          goto LABEL_4;
      }
      return result;
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
      *((_DWORD *)a2 + 19) = 1;
      return result;
    default:
      return v10;
  }
}
