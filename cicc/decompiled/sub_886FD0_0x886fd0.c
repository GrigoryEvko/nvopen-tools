// Function: sub_886FD0
// Address: 0x886fd0
//
void __fastcall sub_886FD0(__int64 a1)
{
  __int64 v1; // rbp
  _QWORD *v2; // r12
  int v3; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v4; // [rsp-8h] [rbp-8h]

  if ( !dword_4F5FFB4 )
  {
    v4 = v1;
    v3 = 0;
    v2 = qword_4D049A0;
    qword_4D049A0[6] = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 24) = v2;
    sub_885620((__int64)v2, dword_4F04C5C, &v3);
    sub_881ED0((__int64)v2, dword_4F04C64, v3);
    dword_4F5FFB4 = 1;
  }
}
