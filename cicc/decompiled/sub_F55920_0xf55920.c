// Function: sub_F55920
// Address: 0xf55920
//
__int64 __fastcall sub_F55920(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 result; // rax
  _BYTE *v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax

  v2 = a1;
  sub_B44570(a1);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v3 = *(_QWORD *)(a1 - 8);
    v2 = v3 + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  }
  else
  {
    v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  }
  for ( result = 0; v2 != v3; v3 += 32 )
  {
    v5 = *(_BYTE **)v3;
    if ( **(_BYTE **)v3 > 0x1Cu )
    {
      v6 = *((_QWORD *)v5 + 1);
      if ( *(_BYTE *)(v6 + 8) != 11 )
      {
        v7 = sub_ACADE0((__int64 **)v6);
        if ( *(_QWORD *)v3 )
        {
          v10 = *(_QWORD *)(v3 + 8);
          **(_QWORD **)(v3 + 16) = v10;
          if ( v10 )
            *(_QWORD *)(v10 + 16) = *(_QWORD *)(v3 + 16);
        }
        *(_QWORD *)v3 = v7;
        if ( v7 )
        {
          v11 = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v3 + 8) = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 16) = v3 + 8;
          *(_QWORD *)(v3 + 16) = v7 + 16;
          *(_QWORD *)(v7 + 16) = v3;
        }
        v12 = *(unsigned int *)(a2 + 8);
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v12 + 1, 8u, v8, v9);
          v12 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v5;
        result = 1;
        ++*(_DWORD *)(a2 + 8);
      }
    }
  }
  return result;
}
