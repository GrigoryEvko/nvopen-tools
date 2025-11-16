// Function: sub_11FAF40
// Address: 0x11faf40
//
__int64 __fastcall sub_11FAF40(int a1, __int64 a2, _DWORD *a3)
{
  _QWORD *v4; // rdi
  int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdi
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-10h]
  __int64 v13; // [rsp+8h] [rbp-8h]

  *a3 = a1;
  if ( a1 )
  {
    if ( a1 == 15 )
    {
      v4 = *(_QWORD **)a2;
      v5 = *(unsigned __int8 *)(a2 + 8);
      if ( (unsigned int)(v5 - 17) > 1 )
      {
        v7 = sub_BCB2A0(v4);
      }
      else
      {
        BYTE4(v13) = (_BYTE)v5 == 18;
        LODWORD(v13) = *(_DWORD *)(a2 + 32);
        v6 = (__int64 *)sub_BCB2A0(v4);
        v7 = sub_BCE1B0(v6, v13);
      }
      return sub_AD64C0(v7, 1, 0);
    }
    else
    {
      return 0;
    }
  }
  else
  {
    v8 = *(_QWORD **)a2;
    v9 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v9 - 17) > 1 )
    {
      v11 = sub_BCB2A0(v8);
    }
    else
    {
      BYTE4(v12) = (_BYTE)v9 == 18;
      LODWORD(v12) = *(_DWORD *)(a2 + 32);
      v10 = (__int64 *)sub_BCB2A0(v8);
      v11 = sub_BCE1B0(v10, v12);
    }
    return sub_AD64C0(v11, 0, 0);
  }
}
