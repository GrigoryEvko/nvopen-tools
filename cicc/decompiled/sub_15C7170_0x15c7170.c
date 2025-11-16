// Function: sub_15C7170
// Address: 0x15c7170
//
void __fastcall sub_15C7170(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  const void *v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  size_t v9; // r14
  __int64 v10; // r14
  unsigned int v11; // eax
  __int64 v12; // rax
  _DWORD *v13; // rdx
  _WORD *v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // r14
  unsigned int v17; // eax
  _QWORD v18[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( !*a1 )
    return;
  v3 = sub_15C70D0((__int64)a1);
  v4 = v3;
  if ( *(_BYTE *)v3 == 15 || (v3 = *(_QWORD *)(v3 - 8LL * *(unsigned int *)(v3 + 8)), (v4 = v3) != 0) )
  {
    v5 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v3 + 8));
    if ( v5 )
    {
      v6 = (const void *)sub_161E970(v5);
      v8 = *(_BYTE **)(a2 + 24);
      v9 = v7;
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v8 >= v7 )
      {
        if ( v7 )
        {
          memcpy(v8, v6, v7);
          v8 = (_BYTE *)(v9 + *(_QWORD *)(a2 + 24));
          *(_QWORD *)(a2 + 24) = v8;
        }
        goto LABEL_8;
      }
      sub_16E7EE0(a2, (const char *)v6);
    }
  }
  v8 = *(_BYTE **)(a2 + 24);
LABEL_8:
  if ( *(_QWORD *)(a2 + 16) <= (unsigned __int64)v8 )
  {
    v10 = sub_16E7DE0(a2, 58);
  }
  else
  {
    v10 = a2;
    *(_QWORD *)(a2 + 24) = v8 + 1;
    *v8 = 58;
  }
  v11 = sub_15C70B0((__int64)a1);
  sub_16E7A90(v10, v11);
  if ( (unsigned int)sub_15C70C0((__int64)a1) )
  {
    v15 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(a2 + 16) )
    {
      v16 = sub_16E7DE0(a2, 58);
    }
    else
    {
      v16 = a2;
      *(_QWORD *)(a2 + 24) = v15 + 1;
      *v15 = 58;
    }
    v17 = sub_15C70C0((__int64)a1);
    sub_16E7A90(v16, v17);
  }
  v12 = sub_15C70F0((__int64)a1);
  sub_15C7080(v18, v12);
  if ( v18[0] )
  {
    v13 = *(_DWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v13 <= 3u )
    {
      sub_16E7EE0(a2, " @[ ", 4);
    }
    else
    {
      *v13 = 542851104;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    sub_15C7170(v18, a2);
    v14 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v14 <= 1u )
    {
      sub_16E7EE0(a2, " ]", 2);
    }
    else
    {
      *v14 = 23840;
      *(_QWORD *)(a2 + 24) += 2LL;
    }
    if ( v18[0] )
      sub_161E7C0(v18);
  }
}
