// Function: sub_1D0DFF0
// Address: 0x1d0dff0
//
void __fastcall sub_1D0DFF0(__int64 a1)
{
  __int64 v2; // rdi
  char v3; // al
  unsigned int v4; // esi
  int v5; // eax
  unsigned int *v6; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    while ( *(_DWORD *)(a1 + 20) <= *(_DWORD *)(a1 + 16) )
    {
LABEL_7:
      v5 = *(_DWORD *)(v2 + 56);
      if ( !v5
        || (v6 = (unsigned int *)(*(_QWORD *)(v2 + 32) + 40LL * (unsigned int)(v5 - 1)),
            *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]) != 111) )
      {
        *(_QWORD *)(a1 + 8) = 0;
        return;
      }
      *(_QWORD *)(a1 + 8) = *(_QWORD *)v6;
      sub_1D0DF70(a1);
      v2 = *(_QWORD *)(a1 + 8);
      if ( !v2 )
        return;
    }
    while ( !(unsigned __int8)sub_1D18C40(v2) )
    {
      v2 = *(_QWORD *)(a1 + 8);
      v4 = *(_DWORD *)(a1 + 16) + 1;
      *(_DWORD *)(a1 + 16) = v4;
      if ( *(_DWORD *)(a1 + 20) <= v4 )
        goto LABEL_7;
    }
    v3 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL) + 16LL * (unsigned int)(*(_DWORD *)(a1 + 16))++);
    *(_BYTE *)(a1 + 24) = v3;
  }
}
