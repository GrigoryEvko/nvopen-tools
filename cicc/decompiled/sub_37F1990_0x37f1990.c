// Function: sub_37F1990
// Address: 0x37f1990
//
void __fastcall sub_37F1990(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned int v3; // edx
  __int64 v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13[7]; // [rsp-38h] [rbp-38h] BYREF

  v2 = *(unsigned __int16 *)(a2 + 6);
  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 56) + 4 * v2);
  if ( (_WORD)v2 )
  {
    if ( v3 )
    {
      v5 = *(_QWORD *)(a1 + 8);
      v13[0] = *(_QWORD *)(a1 + 40);
      v13[1] = v3;
      v6 = sub_37F0BD0(v5, v13);
      v7 = v6;
      if ( v6 != (_QWORD *)(v5 + 8) )
      {
        v8 = *(_QWORD **)(a1 + 24);
        if ( v8 && *(_BYTE *)(a1 + 48) )
        {
          v9 = *((unsigned int *)v6 + 14);
          v10 = (_QWORD *)(*v8 + 16 * v9);
          v11 = v10[1];
          if ( v11 )
          {
            LODWORD(v12) = *((_DWORD *)v6 + 14);
            do
              v12 = (unsigned int)(v12 + 1);
            while ( *(_QWORD *)(*v8 + 16 * v12 + 8) );
            v11 = (unsigned int)(v12 - v9);
          }
          sub_37F1600(v8, v10, v11);
        }
        *(_QWORD *)(a1 + 40) = v7[6];
      }
    }
  }
}
