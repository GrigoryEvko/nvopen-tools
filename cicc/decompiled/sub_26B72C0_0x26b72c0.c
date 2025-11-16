// Function: sub_26B72C0
// Address: 0x26b72c0
//
void __fastcall sub_26B72C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // rax

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 != a1 + 72 )
  {
    while ( 1 )
    {
      v4 = v3 - 24;
      if ( !v3 )
        v4 = 0;
      if ( sub_AA4E50(v4) )
        break;
      v7 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v7 == v4 + 48 )
        goto LABEL_17;
      if ( !v7 )
        BUG();
      v8 = v7 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_17:
        BUG();
      if ( *(_BYTE *)(v7 - 24) == 30
        && (unsigned int)**(unsigned __int8 **)(v7 - 32LL * (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) - 24) - 12 > 1 )
      {
        v9 = *(unsigned int *)(a2 + 8);
        if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 8u, v5, v6);
          v9 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v8;
        ++*(_DWORD *)(a2 + 8);
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
      else
      {
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
    }
  }
}
