// Function: sub_D03760
// Address: 0xd03760
//
void __fastcall sub_D03760(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // rbx
  __int64 v3; // rsi
  _QWORD *v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // r13

  v1 = *(unsigned int *)(a1 + 80);
  *(_QWORD *)a1 = &unk_49DDC10;
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 64);
    v3 = 2 * v1;
    v4 = &v2[v3];
    do
    {
      if ( *v2 != -8192 && *v2 != -4096 )
      {
        v5 = v2[1];
        if ( v5 )
        {
          if ( (v5 & 4) != 0 )
          {
            v6 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
            v7 = v6;
            if ( v6 )
            {
              if ( (_QWORD *)*v6 != v6 + 2 )
                _libc_free(*v6, v3 * 8);
              v3 = 6;
              j_j___libc_free_0(v7, 48);
            }
          }
        }
      }
      v2 += 2;
    }
    while ( v4 != v2 );
    v1 = *(unsigned int *)(a1 + 80);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 16 * v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 48), 8);
  nullsub_184();
}
