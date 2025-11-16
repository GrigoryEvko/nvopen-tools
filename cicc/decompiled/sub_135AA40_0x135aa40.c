// Function: sub_135AA40
// Address: 0x135aa40
//
__int64 __fastcall sub_135AA40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax

  if ( *(_BYTE *)(a2 + 16) != 78
    || (v3 = *(_QWORD *)(a2 - 24), *(_BYTE *)(v3 + 16))
    || (*(_BYTE *)(v3 + 33) & 0x20) == 0
    || (result = (unsigned int)(*(_DWORD *)(v3 + 36) - 35), (unsigned int)result > 3)
    && ((*(_BYTE *)(v3 + 33) & 0x20) == 0
     || (result = *(unsigned int *)(v3 + 36), (_DWORD)result != 4) && (_DWORD)result != 191) )
  {
    if ( (unsigned __int8)sub_15F2ED0(a2) || (result = sub_15F3040(a2), (_BYTE)result) )
    {
      result = sub_1C30710(a2);
      if ( !(_BYTE)result )
      {
        v5 = sub_1358E20(a1, a2);
        if ( !v5 )
        {
          v6 = sub_22077B0(72);
          if ( v6 )
          {
            *(_QWORD *)(v6 + 16) = 0;
            v7 = 0;
            *(_QWORD *)(v6 + 24) = v6 + 16;
            *(_QWORD *)(v6 + 32) = 0;
            *(_QWORD *)(v6 + 40) = 0;
            *(_QWORD *)(v6 + 48) = 0;
            *(_QWORD *)(v6 + 56) = 0;
            *(_QWORD *)(v6 + 64) = 0;
          }
          else
          {
            v7 = MEMORY[0] & 7;
          }
          v8 = *(_QWORD *)(a1 + 8);
          *(_QWORD *)(v6 + 8) = a1 + 8;
          v8 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v6 = v8 | v7;
          *(_QWORD *)(v8 + 8) = v6;
          v9 = *(_QWORD *)(a1 + 8) & 7LL | v6;
          *(_QWORD *)(a1 + 8) = v9;
          v5 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        }
        return sub_135A960(v5, a2);
      }
    }
  }
  return result;
}
