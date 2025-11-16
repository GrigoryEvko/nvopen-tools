// Function: sub_C52060
// Address: 0xc52060
//
void __fastcall sub_C52060(_QWORD *a1, __int64 a2, __int64 a3, const void *a4, size_t a5)
{
  __int64 v8; // r15
  unsigned int v9; // eax
  unsigned int v10; // r8d
  _QWORD *v11; // r10
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // r10
  _QWORD *v19; // rcx
  _QWORD *v20; // [rsp+8h] [rbp-68h]
  _QWORD *v21; // [rsp+10h] [rbp-60h]
  unsigned int v22; // [rsp+18h] [rbp-58h]

  if ( !*(_QWORD *)(a2 + 32) )
  {
    v8 = a3 + 128;
    v9 = sub_C92610(a4, a5);
    v10 = sub_C92740(v8, a4, a5, v9);
    v11 = (_QWORD *)(*(_QWORD *)(a3 + 128) + 8LL * v10);
    if ( *v11 )
    {
      if ( *v11 != -8 )
      {
        v12 = sub_CB72A0(v8, a4);
        v13 = sub_CB6200(v12, *a1, a1[1]);
        v14 = sub_904010(v13, ": CommandLine Error: Option '");
        v15 = sub_A51340(v14, a4, a5);
        sub_904010(v15, "' registered more than once!\n");
        sub_C64ED0("inconsistency in registered CommandLine options", 1);
      }
      --*(_DWORD *)(a3 + 144);
    }
    v21 = v11;
    v22 = v10;
    v16 = sub_C7D670(a5 + 17, 8);
    v17 = v22;
    v18 = v21;
    v19 = (_QWORD *)v16;
    if ( a5 )
    {
      v20 = (_QWORD *)v16;
      memcpy((void *)(v16 + 16), a4, a5);
      v19 = v20;
      v18 = v21;
      v17 = v22;
    }
    *((_BYTE *)v19 + a5 + 16) = 0;
    *v19 = a5;
    v19[1] = a2;
    *v18 = v19;
    ++*(_DWORD *)(a3 + 140);
    sub_C929D0(v8, v17);
  }
}
