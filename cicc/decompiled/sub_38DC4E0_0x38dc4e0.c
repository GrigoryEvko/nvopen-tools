// Function: sub_38DC4E0
// Address: 0x38dc4e0
//
void (*__fastcall sub_38DC4E0(__int64 a1, __int64 a2, unsigned __int64 a3))()
{
  __int64 v4; // rax
  __int64 v5; // rdi
  void (*result)(); // rax
  char v7; // dl
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  const char *v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]
  char v14; // [rsp+11h] [rbp-2Fh]

  v4 = *(_QWORD *)a2;
  if ( (*(_BYTE *)(a2 + 8) & 2) != 0 )
  {
    v7 = *(_BYTE *)(a2 + 9);
    if ( (v7 & 0xC) == 8 )
    {
      v7 &= 0xF3u;
      *(_QWORD *)(a2 + 24) = 0;
      *(_BYTE *)(a2 + 9) = v7;
    }
    v4 &= 7u;
    *(_BYTE *)(a2 + 8) &= ~2u;
    *(_QWORD *)a2 = v4;
  }
  else
  {
    if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
LABEL_3:
      v5 = *(_QWORD *)(a1 + 8);
      v14 = 1;
      v13 = 3;
      v12 = "invalid symbol redefinition";
      return (void (*)())sub_38BE3D0(v5, a3, (__int64)&v12);
    }
    v7 = *(_BYTE *)(a2 + 9);
  }
  if ( (v7 & 0xC) == 8 )
  {
    *(_BYTE *)(a2 + 8) |= 4u;
    v11 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
    v4 = v11 | *(_QWORD *)a2 & 7LL;
    *(_QWORD *)a2 = v4;
    if ( v11 || (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
      goto LABEL_3;
  }
  v8 = *(unsigned int *)(a1 + 120);
  v9 = 0;
  if ( (_DWORD)v8 )
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v8 - 32);
  result = (void (*)())((v9 + 48) | v4 & 7);
  *(_QWORD *)a2 = result;
  v10 = *(_QWORD *)(a1 + 16);
  if ( v10 )
  {
    result = *(void (**)())(*(_QWORD *)v10 + 16LL);
    if ( result != nullsub_1937 )
      return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v10, a2);
  }
  return result;
}
