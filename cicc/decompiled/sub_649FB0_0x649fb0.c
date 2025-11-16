// Function: sub_649FB0
// Address: 0x649fb0
//
__int64 __fastcall sub_649FB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // dl
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rdi
  int i; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r14
  __int64 v13; // [rsp-30h] [rbp-30h] BYREF

  result = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v3 = *(_BYTE *)(result + 80);
    if ( v3 == 9 || v3 == 7 )
    {
      v4 = *(_QWORD *)(result + 88);
    }
    else
    {
      if ( v3 != 21 )
        return result;
      result = *(_QWORD *)(result + 88);
      v4 = *(_QWORD *)(result + 192);
    }
    if ( dword_4F077C4 == 2 )
    {
      if ( !v4 )
        return result;
      if ( (*(_BYTE *)(a1 + 177) & 2) == 0 && (unsigned int)sub_6EA1E0(v4) )
      {
        v13 = sub_724DC0(v4, a2, v8, v9, v10, v11);
        v12 = sub_740200(v4);
        if ( v12 )
        {
          if ( (!dword_4D04964 || *(char *)(a1 + 178) >= 0)
            && (!(unsigned int)sub_8D32E0(*(_QWORD *)(v4 + 120))
             || (*(_BYTE *)(v4 + 172) & 8) != 0
             || sub_718E10(v12, v13)) )
          {
            *(_BYTE *)(v4 + 176) |= 2u;
          }
          if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) && *(_BYTE *)(v12 + 173) == 1 )
            *(_BYTE *)(v12 + 169) |= 4u;
        }
        if ( (*(_BYTE *)(v4 + 176) & 1) != 0 )
          *(_BYTE *)(v4 + 172) |= 4u;
        sub_724E30(&v13);
      }
    }
    else if ( !v4 )
    {
      return result;
    }
    v5 = *(_QWORD *)(v4 + 120);
    result = *(_BYTE *)(v5 + 140) & 0xFB;
    if ( (*(_BYTE *)(v5 + 140) & 0xFB) == 8 )
    {
      result = sub_8D4C10(v5, dword_4F077C4 != 2);
      if ( (result & 1) != 0 )
      {
        v6 = sub_8D4130(*(_QWORD *)(v4 + 120));
        for ( i = *(unsigned __int8 *)(v6 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v6 + 140) )
          v6 = *(_QWORD *)(v6 + 160);
        result = (unsigned int)(i - 9);
        if ( (unsigned __int8)result > 2u || (result = sub_8D5870(v6), !(_DWORD)result) )
          *(_BYTE *)(v4 + 176) |= 4u;
      }
    }
  }
  return result;
}
