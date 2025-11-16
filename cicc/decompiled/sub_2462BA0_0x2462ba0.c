// Function: sub_2462BA0
// Address: 0x2462ba0
//
__int64 __fastcall sub_2462BA0(__int64 a1, __int64 a2)
{
  int v2; // edx
  bool v3; // zf
  __int64 v4; // rdx
  int v5; // ecx
  int v6; // eax
  __int64 v7; // rdx
  int v8; // ecx
  int v9; // eax
  __int64 v10; // rdx
  int v11; // ecx
  int v12; // eax
  __int64 v13; // rdx
  int v14; // ecx
  __int64 *v15; // rax
  __int64 *v16; // rdi
  __int64 *v17; // rdi
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 v21; // [rsp-30h] [rbp-30h]
  __int64 v22; // [rsp-28h] [rbp-28h]
  __int64 v23; // [rsp-20h] [rbp-20h]
  __int64 v24; // [rsp-18h] [rbp-18h]
  __int64 v25; // [rsp-10h] [rbp-10h]

  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v2 - 17) > 1 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 8) + 96LL);
  v3 = (_BYTE)v2 == 18;
  v4 = *(_QWORD *)(a2 + 24);
  BYTE4(v21) = v3;
  LODWORD(v21) = *(_DWORD *)(a2 + 32);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v5 - 17) > 1 )
    return sub_BCE1B0(*(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL), v21);
  v6 = *(_DWORD *)(v4 + 32);
  v7 = *(_QWORD *)(v4 + 24);
  BYTE4(v22) = (_BYTE)v5 == 18;
  LODWORD(v22) = v6;
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) > 1 )
  {
    v18 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
  }
  else
  {
    v9 = *(_DWORD *)(v7 + 32);
    v10 = *(_QWORD *)(v7 + 24);
    BYTE4(v23) = (_BYTE)v8 == 18;
    LODWORD(v23) = v9;
    v11 = *(unsigned __int8 *)(v10 + 8);
    if ( (unsigned int)(v11 - 17) > 1 )
    {
      v17 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
    }
    else
    {
      v12 = *(_DWORD *)(v10 + 32);
      v13 = *(_QWORD *)(v10 + 24);
      BYTE4(v24) = (_BYTE)v11 == 18;
      LODWORD(v24) = v12;
      v14 = *(unsigned __int8 *)(v13 + 8);
      if ( (unsigned int)(v14 - 17) > 1 )
      {
        v16 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
      }
      else
      {
        BYTE4(v25) = (_BYTE)v14 == 18;
        LODWORD(v25) = *(_DWORD *)(v13 + 32);
        v15 = (__int64 *)sub_2462BA0(a1, *(_QWORD *)(v13 + 24));
        v16 = (__int64 *)sub_BCE1B0(v15, v25);
      }
      v17 = (__int64 *)sub_BCE1B0(v16, v24);
    }
    v18 = (__int64 *)sub_BCE1B0(v17, v23);
  }
  v19 = (__int64 *)sub_BCE1B0(v18, v22);
  return sub_BCE1B0(v19, v21);
}
