// Function: sub_18DE970
// Address: 0x18de970
//
__int64 __fastcall sub_18DE970(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F99CCC);
  if ( !byte_4FAE160 )
  {
    sub_1636A10(a2, (__int64)&unk_4F99CCC);
    result = *(unsigned int *)(a2 + 120);
    if ( *(_DWORD *)(a2 + 124) > (unsigned int)result )
      goto LABEL_3;
LABEL_9:
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    result = *(unsigned int *)(a2 + 120);
    goto LABEL_3;
  }
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F9E06C;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v6;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v6 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4F99CCC;
  result = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = result;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)result )
    goto LABEL_9;
LABEL_3:
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
