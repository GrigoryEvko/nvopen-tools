// Function: sub_134CC90
// Address: 0x134cc90
//
__int64 __fastcall sub_134CC90(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  _QWORD *v3; // rbx
  _QWORD *i; // r14
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rdx
  _QWORD *v18; // rax

  v2 = 63;
  v3 = *(_QWORD **)(a1 + 48);
  for ( i = *(_QWORD **)(a1 + 56); i != v3; ++v3 )
  {
    v2 &= (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v3 + 48LL))(*v3, a2);
    if ( v2 == 4 )
      return 4;
  }
  v6 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 16) == 78 )
  {
    v7 = *(_QWORD *)(v6 - 24);
    if ( !*(_BYTE *)(v7 + 16) && (*(_BYTE *)(v7 + 33) & 0x20) != 0 && v6 )
    {
      v8 = *(_DWORD *)(v7 + 36);
      if ( v8 == 4271 )
      {
        v17 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v18 = *(_QWORD **)(v17 + 24);
        if ( *(_DWORD *)(v17 + 32) > 0x40u )
          v18 = (_QWORD *)*v18;
        if ( ((unsigned __int8)v18 & 1) == 0 )
          return 4;
      }
      else if ( v8 > 0x10AF )
      {
        if ( v8 == 4277 )
        {
          v13 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
          v14 = *(_QWORD **)(v13 + 24);
          if ( *(_DWORD *)(v13 + 32) > 0x40u )
            v14 = (_QWORD *)*v14;
          if ( ((unsigned __int8)v14 & 1) != 0 )
            return 4;
        }
        else if ( v8 == 4350 )
        {
          v11 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
          v12 = *(_QWORD **)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
            v12 = (_QWORD *)*v12;
          if ( (unsigned __int8)sub_1C30500(v12) )
            return 4;
        }
      }
      else if ( v8 == 4057 )
      {
        v15 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v16 = *(_QWORD **)(v15 + 24);
        if ( *(_DWORD *)(v15 + 32) > 0x40u )
          v16 = (_QWORD *)*v16;
        if ( (unsigned __int8)sub_1C278B0(v16) == 6 )
          return 4;
      }
      else if ( v8 == 4085 )
      {
        v9 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v10 = *(_QWORD **)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
          v10 = (_QWORD *)*v10;
        if ( ((unsigned __int16)v10 & 0x1E0) == 0xE0 )
          return 4;
      }
    }
  }
  return v2;
}
