// Function: sub_327C6E0
// Address: 0x327c6e0
//
__int64 __fastcall sub_327C6E0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 result; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rdi
  int v7; // ecx
  char v8; // si
  int v9; // edx
  __int64 v10; // rcx
  int v11; // edx

  v3 = *a2;
  result = a1;
  v5 = a2[1] + a3;
  v6 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 )
  {
    v8 = *((_BYTE *)a2 + 20);
    if ( (v3 & 4) != 0 )
    {
      *(_QWORD *)(result + 8) = v5;
      v9 = *(_DWORD *)(v6 + 12);
      *(_BYTE *)(result + 20) = v8;
      *(_QWORD *)result = v6 | 4;
      *(_DWORD *)(result + 16) = v9;
    }
    else
    {
      v10 = *(_QWORD *)(v6 + 8);
      *(_QWORD *)(result + 8) = v5;
      *(_QWORD *)result = v6;
      v11 = *(unsigned __int8 *)(v10 + 8);
      *(_BYTE *)(result + 20) = v8;
      if ( (unsigned int)(v11 - 17) <= 1 )
        v10 = **(_QWORD **)(v10 + 16);
      *(_DWORD *)(result + 16) = *(_DWORD *)(v10 + 8) >> 8;
    }
  }
  else
  {
    v7 = *((_DWORD *)a2 + 4);
    *(_QWORD *)result = 0;
    *(_QWORD *)(result + 8) = v5;
    *(_DWORD *)(result + 16) = v7;
    *(_BYTE *)(result + 20) = 0;
  }
  return result;
}
