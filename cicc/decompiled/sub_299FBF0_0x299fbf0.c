// Function: sub_299FBF0
// Address: 0x299fbf0
//
_QWORD *__fastcall sub_299FBF0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rcx
  unsigned __int8 v4; // al
  __int64 v5; // rcx
  int v6; // eax
  unsigned int v7; // edx
  __int64 v8; // rcx
  unsigned int v9; // edi
  _QWORD *v11; // rax
  int v12; // [rsp+1Ch] [rbp-34h] BYREF
  unsigned int v13; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+24h] [rbp-2Ch] BYREF
  __int64 v15; // [rsp+28h] [rbp-28h]
  __int64 v16; // [rsp+30h] [rbp-20h]
  __int64 v17; // [rsp+38h] [rbp-18h]

  v2 = a1 - 16;
  v4 = *(_BYTE *)(a1 - 16);
  if ( LOBYTE(qword_4F813A8[8]) )
  {
    if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
      v5 = *(_QWORD *)(a1 - 32);
    else
      v5 = v2 - 8LL * ((v4 >> 2) & 0xF);
    v6 = 0;
    if ( **(_BYTE **)v5 == 20 )
    {
      v7 = *(_DWORD *)(*(_QWORD *)v5 + 4LL);
      v6 = v7 & 7;
      if ( v6 == 7 )
      {
        if ( (v7 & 0xFFFFFFF8) != 0 )
        {
          if ( (v7 & 0x10000000) != 0 )
            v6 = HIWORD(v7) & 7;
          else
            v6 = (unsigned __int16)(v7 >> 3);
        }
      }
      else
      {
        v6 = (unsigned __int8)v7;
      }
    }
    v12 = v6;
    if ( a2 != v6 )
    {
      LOBYTE(v17) = 1;
      return sub_26BDBC0(a1, a2);
    }
LABEL_19:
    v16 = a1;
    LOBYTE(v17) = 1;
    return (_QWORD *)v16;
  }
  if ( (v4 & 2) != 0 )
    v8 = *(_QWORD *)(a1 - 32);
  else
    v8 = v2 - 8LL * ((v4 >> 2) & 0xF);
  v9 = 0;
  if ( **(_BYTE **)v8 == 20 )
    v9 = *(_DWORD *)(*(_QWORD *)v8 + 4LL);
  sub_AF16E0(v9, &v12, (int *)&v13, &v14);
  if ( v12 == a2 )
    goto LABEL_19;
  v15 = sub_AF17B0(a2, v13, v14);
  if ( BYTE4(v15) )
  {
    v11 = sub_26BDBC0(a1, v15);
    LOBYTE(v17) = 1;
    return v11;
  }
  else
  {
    LOBYTE(v17) = 0;
  }
  return (_QWORD *)v16;
}
