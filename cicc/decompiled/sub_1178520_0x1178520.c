// Function: sub_1178520
// Address: 0x1178520
//
__int64 __fastcall sub_1178520(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __int64 result; // rax
  _BYTE *v5; // r13
  __int64 v7; // r12
  unsigned int v9; // eax
  __int64 v10; // r15
  int v11; // edi
  __int64 v12; // rbx
  __int64 v13; // rax
  char v14; // si
  __int64 *v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // [rsp-70h] [rbp-70h]
  _WORD v18[52]; // [rsp-68h] [rbp-68h] BYREF

  result = 0;
  if ( *a2 > 0x1Cu )
  {
    v5 = a3;
    if ( *a3 > 0x1Cu )
    {
      v7 = (__int64)a2;
      v9 = sub_B53110(*(_WORD *)(a1 + 2) & 0x3F);
      v10 = *(_QWORD *)(a1 - 64);
      v11 = v9;
      v12 = *(_QWORD *)(a1 - 32);
      if ( *v5 == 44 && v10 == *((_QWORD *)v5 - 8) && v12 == *((_QWORD *)v5 - 4) )
      {
        v11 = sub_B52F50(v9);
        v7 = (__int64)v5;
        v5 = a2;
      }
      if ( v11 == 38
        && *(_BYTE *)v7 == 44
        && v10 == *(_QWORD *)(v7 - 64)
        && v12 == *(_QWORD *)(v7 - 32)
        && *v5 == 44
        && v12 == *((_QWORD *)v5 - 8)
        && v10 == *((_QWORD *)v5 - 4)
        && (sub_B44900(v7) || sub_B448F0(v7))
        && (sub_B44900((__int64)v5) || sub_B448F0((__int64)v5)) )
      {
        sub_B447F0((unsigned __int8 *)v7, 0);
        if ( !sub_B44900(v7) )
        {
          v13 = *(_QWORD *)(v7 + 16);
          v14 = 0;
          if ( v13 )
            v14 = *(_QWORD *)(v13 + 8) == 0;
          sub_B44850((unsigned __int8 *)v7, v14);
        }
        v15 = *(__int64 **)(a4 + 72);
        v18[16] = 257;
        v16 = sub_ACD6D0(v15);
        return sub_B33C40(a4, 1u, v7, v16, v17, (__int64)v18);
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
