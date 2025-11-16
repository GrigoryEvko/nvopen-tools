// Function: sub_35E6920
// Address: 0x35e6920
//
__int64 __fastcall sub_35E6920(__int64 a1, char a2)
{
  __int64 v2; // r14
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 (*v5)(); // rax
  __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  void (__fastcall *v14)(unsigned __int64); // rax
  unsigned __int64 v15; // rdi
  char v16[32]; // [rsp+10h] [rbp-60h] BYREF
  char v17; // [rsp+30h] [rbp-40h]
  char v18; // [rsp+31h] [rbp-3Fh]

  v2 = *(_QWORD *)(a1 + 8);
  if ( a2 )
  {
    v7 = sub_22077B0(0x690u);
    v8 = v7;
    if ( v7 )
    {
      *(_QWORD *)(v7 + 8) = v2;
      v9 = v7 + 72;
      v18 = 1;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)(v7 + 24) = 0;
      *(_DWORD *)(v7 + 32) = 0;
      *(_WORD *)(v7 + 36) = 0;
      *(_QWORD *)(v7 + 64) = 0x1000000000LL;
      *(_QWORD *)(v7 + 56) = v7 + 72;
      *(_BYTE *)(v7 + 52) = 0;
      *(_QWORD *)v7 = &unk_4A29FE0;
      *(_QWORD *)v16 = "TopQ";
      *(_QWORD *)(v7 + 40) = 0;
      *(_DWORD *)(v7 + 48) = 0;
      *(_QWORD *)(v7 + 136) = 0;
      v17 = 3;
      sub_2EC8790(v7 + 144, 1, v16, (__int64)&unk_4A29FE0);
      *(_QWORD *)v16 = "BotQ";
      v18 = 1;
      v17 = 3;
      sub_2EC8790(v8 + 864, 2, v16, v10);
      *(_WORD *)(v8 + 1668) = 0;
      *(_BYTE *)(v8 + 1584) = 0;
      *(_QWORD *)(v8 + 1588) = 0;
      *(_QWORD *)(v8 + 1600) = 0;
      *(_QWORD *)(v8 + 1608) = 0;
      *(_DWORD *)(v8 + 1616) = 0;
      *(_WORD *)(v8 + 1620) = 0;
      *(_QWORD *)(v8 + 1624) = 0;
      *(_BYTE *)(v8 + 1632) = 0;
      *(_QWORD *)(v8 + 1636) = 0;
      *(_QWORD *)(v8 + 1672) = 0;
      *(_QWORD *)(v8 + 1648) = 0;
      *(_QWORD *)(v8 + 1656) = 0;
      *(_DWORD *)(v8 + 1664) = 0;
      v4 = sub_22077B0(0xDD8u);
      if ( !v4 )
      {
        v14 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v8 + 16LL);
        if ( v14 == sub_2EC8510 )
        {
          *(_QWORD *)v8 = &unk_4A29FE0;
          sub_2EC8240(v8 + 864);
          sub_2EC8240(v8 + 144);
          v15 = *(_QWORD *)(v8 + 56);
          if ( v15 != v9 )
            _libc_free(v15);
          j_j___libc_free_0(v8);
        }
        else
        {
          ((void (__fastcall *)(unsigned __int64, __int64, __int64, void *, unsigned __int64))v14)(
            v8,
            2,
            v11,
            &unk_4A29FE0,
            v8 + 864);
        }
        return v4;
      }
    }
    else
    {
      v4 = sub_22077B0(0xDD8u);
      if ( !v4 )
        return v4;
    }
    v12 = *(_QWORD *)(a1 + 8);
    sub_2F91670(v4, *(__int64 **)(v12 + 8), *(_QWORD *)(v12 + 16), 1);
    *(_QWORD *)(v4 + 3520) = 0;
    *(_QWORD *)(v4 + 3528) = 0;
    *(_DWORD *)(v4 + 3536) = 0;
    *(_QWORD *)v4 = &unk_4A29DA8;
    *(_QWORD *)(v4 + 3456) = *(_QWORD *)(v12 + 40);
    v13 = *(_QWORD *)(v12 + 48);
    *(_QWORD *)(v4 + 3472) = v8;
    *(_QWORD *)(v4 + 3464) = v13;
    *(_QWORD *)(v4 + 3480) = 0;
    *(_QWORD *)(v4 + 3488) = 0;
    *(_QWORD *)(v4 + 3496) = 0;
    *(_QWORD *)(v4 + 3504) = 0;
    *(_QWORD *)(v4 + 3512) = 0;
    return v4;
  }
  v3 = *(_QWORD *)(v2 + 32);
  v4 = 0;
  v5 = *(__int64 (**)())(*(_QWORD *)v3 + 40LL);
  if ( v5 == sub_23CE2A0 )
    return v4;
  return ((__int64 (__fastcall *)(__int64, __int64))v5)(v3, v2);
}
