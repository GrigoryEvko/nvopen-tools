// Function: sub_68A480
// Address: 0x68a480
//
__int64 __fastcall sub_68A480(__int64 a1, __int64 *a2, _QWORD **a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 result; // rax
  unsigned int v10; // eax
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // [rsp-10h] [rbp-1C0h]
  unsigned int v15; // [rsp+4h] [rbp-1ACh]
  __int64 v16; // [rsp+8h] [rbp-1A8h]
  int v17; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v18; // [rsp+18h] [rbp-198h] BYREF
  _BYTE v19[400]; // [rsp+20h] [rbp-190h] BYREF

  v16 = *(_QWORD *)(sub_6E3DA0(a1, 0) + 128);
  v6 = *(_DWORD *)(a4 + 40);
  if ( (v6 & 0x2000) == 0 )
  {
    v7 = *(_QWORD *)(a4 + 48);
    if ( !*(_DWORD *)(v7 + 84) || *(_BYTE *)(v16 + 63) || (v6 & 0x40000) != 0 )
    {
      v10 = *(_DWORD *)(v7 + 92);
      *(_DWORD *)(v7 + 92) = 0;
      v15 = v10;
      v11 = sub_869530(
              v16,
              *(_QWORD *)(a4 + 32),
              *(_QWORD *)(a4 + 24),
              (unsigned int)&v18,
              *(_DWORD *)(a4 + 40),
              *(_QWORD *)(a4 + 48),
              (__int64)&v17);
      v12 = v14;
      if ( v17 )
        *(_BYTE *)(a4 + 56) = 1;
      if ( v11 )
      {
        *(_DWORD *)(v7 + 92) = v15;
        do
        {
          v13 = sub_6F8A60(a1, a4, v12);
          if ( *a2 )
            **a3 = v13;
          else
            *a2 = v13;
          *a3 = (_QWORD *)v13;
          sub_867630(v18, 0);
        }
        while ( (unsigned int)sub_866C00(v18) );
      }
      else if ( *(_DWORD *)(*(_QWORD *)(a4 + 48) + 92LL) )
      {
        result = *(unsigned int *)(a4 + 40);
        *(_DWORD *)(v7 + 92) = v15;
        if ( (result & 0x4000) != 0 )
          goto LABEL_5;
LABEL_19:
        if ( (*(_BYTE *)(a4 + 40) & 0x40) == 0 )
          return result;
        result = v16;
        if ( *(_BYTE *)(v16 + 62) )
          return result;
        goto LABEL_5;
      }
      result = v15;
      *(_DWORD *)(v7 + 92) = v15;
      goto LABEL_19;
    }
  }
LABEL_5:
  v8 = sub_73B8B0(a1, 0x4000);
  *(_BYTE *)(v8 + 26) |= 4u;
  sub_6E70E0(v8, v19);
  result = sub_6E3060(v19);
  *(_QWORD *)(result + 16) = v16;
  if ( *a2 )
  {
    *(_BYTE *)(result + 10) = ((*(_DWORD *)(a4 + 40) & 0x40) != 0) | *(_BYTE *)(result + 10) & 0xFE;
    **a3 = result;
  }
  else
  {
    *a2 = result;
  }
  *a3 = (_QWORD *)result;
  return result;
}
