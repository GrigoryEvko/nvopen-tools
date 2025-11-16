// Function: sub_37498B0
// Address: 0x37498b0
//
__int64 __fastcall sub_37498B0(__int64 *a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // r12d
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned int v8; // ebx
  __int64 v9; // rdx
  __int64 v11; // rax
  __int64 *v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // r8
  __int64 (*v15)(); // rax
  unsigned int v16; // edx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = a1[16];
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v8 = sub_2D5BAE0(v6, a1[14], *(__int64 **)(*(_QWORD *)v7 + 8LL), 0);
  v9 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(a2 + 8), 0);
  LOBYTE(v3) = (unsigned __int16)v8 <= 1u || (_WORD)v9 == 1 || (_WORD)v9 == 0;
  if ( !(_BYTE)v3 )
  {
    v11 = a1[16];
    if ( *(_QWORD *)(v11 + 8LL * (unsigned __int16)v9 + 112) )
    {
      if ( *(_QWORD *)(v11 + 8LL * (unsigned __int16)v8 + 112) )
      {
        v12 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
            ? *(__int64 **)(a2 - 8)
            : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v17 = v9;
        v13 = sub_3746830(a1, *v12);
        v14 = v13;
        if ( v13 )
        {
          v15 = *(__int64 (**)())(*a1 + 64);
          if ( v15 == sub_3740EE0 )
            return v3;
          v16 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64, _QWORD, __int64))v15)(a1, v8, v17, a3, v14);
          if ( v16 )
          {
            v3 = 1;
            sub_3742B00((__int64)a1, (_BYTE *)a2, v16, 1);
            return v3;
          }
        }
      }
    }
  }
  return 0;
}
