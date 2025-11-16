// Function: sub_887050
// Address: 0x887050
//
void __fastcall sub_887050(__int64 a1)
{
  __int64 v1; // rbp
  __int64 v2; // r12
  int v3; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v4; // [rsp-8h] [rbp-8h]

  if ( !dword_4F5FFB0 )
  {
    v4 = v1;
    v3 = 0;
    v2 = qword_4D049B0;
    *(_QWORD *)(qword_4D049B0 + 48LL) = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 24) = v2;
    sub_885620(v2, dword_4F04C5C, &v3);
    sub_881ED0(v2, dword_4F04C64, v3);
    dword_4F5FFB0 = 1;
  }
}
