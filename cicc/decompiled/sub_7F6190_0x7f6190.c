// Function: sub_7F6190
// Address: 0x7f6190
//
_BYTE *__fastcall sub_7F6190(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  _BYTE *v8; // r13
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rbx
  unsigned __int8 v12; // si
  char v13; // al
  __int64 v14; // rdi
  _QWORD *v15; // rax
  unsigned __int8 v16; // si
  __int64 v17; // rdi
  __int64 v19; // rax
  int v20; // [rsp+14h] [rbp-2Ch] BYREF
  int v21; // [rsp+18h] [rbp-28h] BYREF
  _DWORD v22[9]; // [rsp+1Ch] [rbp-24h] BYREF

  sub_87ADD0(a1, &v20, &v21, v22);
  if ( v22[0] )
  {
    v5 = sub_8865A0("destroying_delete_t");
    v6 = sub_7E7CB0(*(_QWORD *)(v5 + 88));
    v7 = sub_72D2E0(*(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL));
    v8 = sub_73E130(a3, v7);
    v9 = sub_73E830(v6);
    *((_QWORD *)v8 + 2) = v9;
    v11 = v9;
  }
  else
  {
    v19 = sub_7E1C10();
    v11 = sub_73E130(a3, v19);
    v8 = v11;
  }
  if ( v20 )
  {
    v12 = byte_4F06A51[0];
    v13 = *(_BYTE *)(a2 + 140);
    if ( v13 == 12 )
    {
      v12 = byte_4F06A51[0];
      v14 = sub_8D4A00(a2);
    }
    else if ( dword_4F077C0 && (v13 == 1 || v13 == 7) )
    {
      v14 = 1;
    }
    else
    {
      v14 = *(_QWORD *)(a2 + 128);
    }
    v15 = sub_73A8E0(v14, v12);
    *((_QWORD *)v11 + 2) = v15;
    v11 = v15;
  }
  if ( v21 )
  {
    v16 = byte_4F06A51[0];
    if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    {
      v16 = byte_4F06A51[0];
      v17 = (unsigned int)sub_8D4AB0(a2, byte_4F06A51[0], v10);
    }
    else
    {
      v17 = *(unsigned int *)(a2 + 136);
    }
    *((_QWORD *)v11 + 2) = sub_73A8E0(v17, v16);
  }
  return v8;
}
