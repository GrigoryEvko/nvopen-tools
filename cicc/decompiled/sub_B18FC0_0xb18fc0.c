// Function: sub_B18FC0
// Address: 0xb18fc0
//
__int64 __fastcall sub_B18FC0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a1 + 200);
  result = *(unsigned int *)(a1 + 208);
  v5 = v3 + 8 * result;
  while ( v3 != v5 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v5 - 8);
      v5 -= 8;
      if ( !v6 )
        break;
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 != v6 + 40 )
        _libc_free(v7, a2);
      a2 = 80;
      result = j_j___libc_free_0(v6, 80);
      if ( v3 == v5 )
        goto LABEL_7;
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 292) = 0;
  return result;
}
