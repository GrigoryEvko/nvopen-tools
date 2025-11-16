// Function: sub_1C30530
// Address: 0x1c30530
//
__int64 __fastcall sub_1C30530(int a1)
{
  unsigned int v1; // r12d
  int v2; // eax
  __int64 v4; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v5; // [rsp+8h] [rbp-28h]
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF

  v1 = 0;
  sub_15E1070(&v4, a1, 0, 0);
  if ( v5 > 0xC )
  {
    if ( *(_QWORD *)v4 == 0x76766E2E6D766C6CLL && *(_DWORD *)(v4 + 8) == 1702112877 && *(_BYTE *)(v4 + 12) == 120
      || v5 != 13
      && (*(_QWORD *)v4 == 0x76766E2E6D766C6CLL && *(_DWORD *)(v4 + 8) == 1819553389 && *(_WORD *)(v4 + 12) == 13412
       || *(_QWORD *)v4 == 0x76766E2E6D766C6CLL && *(_DWORD *)(v4 + 8) == 1970482797 && *(_WORD *)(v4 + 12) == 25708)
      || *(_QWORD *)v4 == 0x76766E2E6D766C6CLL && *(_DWORD *)(v4 + 8) == 2020879981 && *(_BYTE *)(v4 + 12) == 113
      || *(_QWORD *)v4 == 0x76766E2E6D766C6CLL && *(_DWORD *)(v4 + 8) == 1970482797 && *(_BYTE *)(v4 + 12) == 113 )
    {
      goto LABEL_18;
    }
    if ( v5 <= 0x10 )
    {
      if ( v5 <= 0xD )
      {
        v1 = 0;
        goto LABEL_13;
      }
    }
    else if ( !(*(_QWORD *)v4 ^ 0x76766E2E6D766C6CLL | *(_QWORD *)(v4 + 8) ^ 0x6570797473692E6DLL)
           && *(_BYTE *)(v4 + 16) == 112 )
    {
LABEL_18:
      v1 = 1;
      goto LABEL_13;
    }
    if ( *(_QWORD *)v4 != 0x76766E2E6D766C6CLL
      || *(_DWORD *)(v4 + 8) != 1970482797
      || (v2 = 0, *(_WORD *)(v4 + 12) != 29811) )
    {
      v2 = 1;
    }
    LOBYTE(v1) = v2 == 0;
  }
LABEL_13:
  if ( (__int64 *)v4 != &v6 )
    j_j___libc_free_0(v4, v6 + 1);
  return v1;
}
