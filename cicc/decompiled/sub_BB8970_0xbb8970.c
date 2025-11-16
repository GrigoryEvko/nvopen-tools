// Function: sub_BB8970
// Address: 0xbb8970
//
void __fastcall sub_BB8970(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax

  if ( *(_BYTE *)(a2 + 40) )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = *(_QWORD *)(a2 + 32);
    v4 = *(unsigned int *)(v2 + 8);
    if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 12) )
    {
      sub_C8D5F0(*(_QWORD *)(a1 + 8), v2 + 16, v4 + 1, 8);
      v4 = *(unsigned int *)(v2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v2 + 8 * v4) = v3;
    ++*(_DWORD *)(v2 + 8);
  }
}
