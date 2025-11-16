// Function: sub_2A65590
// Address: 0x2a65590
//
void __fastcall sub_2A65590(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rsi
  _QWORD *v6; // rdi
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rax

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  v4 = *a1;
  if ( v3 != a2 + 72 )
  {
    do
    {
      v5 = v3 - 24;
      if ( !v3 )
        v5 = 0;
      if ( *(_BYTE *)(v4 + 68) )
      {
        v6 = *(_QWORD **)(v4 + 48);
        v7 = &v6[*(unsigned int *)(v4 + 60)];
        v8 = v6;
        if ( v6 != v7 )
        {
          while ( v5 != *v8 )
          {
            if ( v7 == ++v8 )
              goto LABEL_10;
          }
          v9 = (unsigned int)(*(_DWORD *)(v4 + 60) - 1);
          *(_DWORD *)(v4 + 60) = v9;
          *v8 = v6[v9];
          ++*(_QWORD *)(v4 + 40);
        }
      }
      else
      {
        v10 = sub_C8CA60(v4 + 40, v5);
        if ( v10 )
        {
          *v10 = -2;
          ++*(_DWORD *)(v4 + 64);
          ++*(_QWORD *)(v4 + 40);
        }
      }
LABEL_10:
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
}
