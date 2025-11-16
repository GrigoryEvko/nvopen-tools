// Function: sub_37D0D70
// Address: 0x37d0d70
//
void __fastcall sub_37D0D70(__int64 a1, unsigned int a2, unsigned __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // rax
  __int64 v6; // rcx
  unsigned int v8; // r11d
  int *v9; // rdx
  int v10; // r10d
  int v11; // edx
  int v12; // r12d

  v4 = *(unsigned int *)(a1 + 3432);
  v6 = *(_QWORD *)(a1 + 3416);
  if ( (_DWORD)v4 )
  {
    v8 = a2 & (v4 - 1);
    v9 = (int *)(v6 + 88LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      if ( v9 != (int *)(v6 + 88 * v4) )
        sub_37CFFC0(a1, a2, *(_QWORD *)(*(_QWORD *)(a1 + 3136) + 8LL * a2), a3, a4);
    }
    else
    {
      v11 = 1;
      while ( v10 != -1 )
      {
        v12 = v11 + 1;
        v8 = (v4 - 1) & (v11 + v8);
        v9 = (int *)(v6 + 88LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
}
