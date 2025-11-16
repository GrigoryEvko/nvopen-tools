// Function: sub_108AF10
// Address: 0x108af10
//
void __fastcall sub_108AF10(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned int v3; // ecx
  __int64 v4; // rax
  __int64 i; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v9; // rax

  v3 = *(_DWORD *)(a2 + 48);
  if ( v3 )
  {
    *(_QWORD *)(a2 + 40) = *a3;
    if ( *(_BYTE *)(a1[23] + 8LL) )
    {
      v4 = 14;
    }
    else
    {
      v4 = 10;
      if ( v3 == 0xFFFF )
      {
        v7 = a1[227];
        v8 = a1[228];
        for ( i = 0; v8 != v7; v7 += 64 )
        {
          if ( *(_DWORD *)(v7 + 48) == *(__int16 *)(a2 + 56) )
          {
            v9 = *(_QWORD *)(v7 + 16);
            *(_QWORD *)(v7 + 40) = *(_QWORD *)(a2 + 40);
            i = 10 * v9;
          }
        }
        goto LABEL_5;
      }
    }
    i = v3 * v4;
LABEL_5:
    v6 = *a3 + i;
    *a3 = v6;
    if ( v6 > a1[30] )
      sub_C64ED0("Relocation data overflowed this object file.", 1u);
  }
}
